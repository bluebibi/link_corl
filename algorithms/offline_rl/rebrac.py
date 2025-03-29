# ReBRAC
# 1. Deeper Networks
# 2. LayerNorm
# 3. Larger Batches
import copy
import os
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from algorithms.offline_rl.a_common import preliminary, soft_update, train_and_eval_loop
from algorithms.offline_rl.b_data_buffer import TensorBatch


@dataclass
class TrainConfig:
    # device: str = "cuda"
    device: str = "cpu"
    env: str = "halfcheetah-medium-expert-v2"  # OpenAI gym environment name
    minari_dataset_name: str = "mujoco/halfcheetah/expert-v0"

    seed: int = 0  # PyTorch and Numpy seeds
    train_seed: int = 10   # training env - random seed
    eval_seed: int = 10   # eval env - random seed

    # evaluation frequency, will evaluate every eval_freq training steps
    eval_freq: int = int(5e3)
    # number of episodes to run during evaluation
    n_episodes: int = 10
    # wandb project name
    project: str = "CORL"
    # wandb group name
    group: str = "ReBRAC-MINARI"
    # wandb run name
    name: str = "ReBRAC"
    # training dataset and evaluation environment

    # coefficient for penalty 1 in ReBRAC
    beta1: float = 0.01
    # coefficient for penalty 1 in ReBRAC
    beta2: float = 0.01

    # discount factor
    gamma: float = 0.99
    # standard deviation for the gaussian exploration noise
    expl_noise: float = 0.1
    # coefficient for the target critic Polyak's update
    tau: float = 0.005
    # scalig coefficient for the noise added to
    # target actor during critic update
    policy_noise: float = 0.2
    # range for the target actor noise clipping
    noise_clip: float = 0.5
    # actor update delay
    policy_freq: int = 2
    # total gradient updates during training
    max_timesteps: int = int(1e6)
    # maximum size of the replay buffer
    buffer_size: int = 2_000_000
    # training batch size
    batch_size: int = 1_024
    # whether to normalize states
    normalize: bool = True
    # whether to normalize reward (like in IQL)
    normalize_reward: bool = False
    reward_scale: float = 5.0  # Reward scale for normalization
    reward_bias: float = -1.0  # Reward bias for normalization
    # path for checkpoints saving, optional
    checkpoints_path: Optional[str] = None
    # file name for loading a model, optional
    load_model: str = ""

    #ReBRAC
    gamma_start = 0.99
    gamma_end = 0.999

    def __post_init__(self):
        self.name = f"{self.name}-{self.env}-{str(uuid.uuid4())[:8]}"
        if self.checkpoints_path is not None:
            self.checkpoints_path = os.path.join(self.checkpoints_path, self.name)



class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, max_action: float):
        super(Actor, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),  # ReBRAC
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh(),
        )

        self.max_action = max_action

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.max_action * self.net(state)

    @torch.no_grad()
    def act(self, state: np.ndarray, device: str = "cpu") -> np.ndarray:
        state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        return self(state).cpu().data.numpy().flatten()


class Critic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super(Critic, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.LayerNorm(normalized_shape=256),                     # ReBRAC
            nn.ReLU(),
            nn.Linear(256, 256),              # ReBRAC
            nn.LayerNorm(normalized_shape=256),                     # ReBRAC
            nn.ReLU(),
            nn.Linear(256, 256),              # ReBRAC
            nn.LayerNorm(normalized_shape=256),                     # ReBRAC
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        sa = torch.cat([state, action], 1)
        return self.net(sa)


class ReBRAC:
    def __init__(
        self,
        max_action: float,
        actor: nn.Module,
        actor_optimizer: torch.optim.Optimizer,
        critic_1: nn.Module,
        critic_1_optimizer: torch.optim.Optimizer,
        critic_2: nn.Module,
        critic_2_optimizer: torch.optim.Optimizer,
        gamma: float = 0.99,
        tau: float = 0.005,
        policy_noise: float = 0.2,
        noise_clip: float = 0.5,
        policy_freq: int = 2,
        beta1: float = 0.01,    # ReBRAC
        beta2: float = 0.01,    # ReBRAC
        device: str = "cpu",
    ):
        self.actor = actor
        self.actor_target = copy.deepcopy(actor)
        self.actor_optimizer = actor_optimizer
        self.critic_1 = critic_1
        self.critic_1_target = copy.deepcopy(critic_1)
        self.critic_1_optimizer = critic_1_optimizer
        self.critic_2 = critic_2
        self.critic_2_target = copy.deepcopy(critic_2)
        self.critic_2_optimizer = critic_2_optimizer

        self.max_action = max_action
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.beta1 = beta1
        self.beta2 = beta2
        self.total_it = 0
        self.device = device

    def train(self, batch: TensorBatch) -> Dict[str, float]:
        log_dict = {}
        self.total_it += 1

        assert len(batch) == 6
        state, action, reward, next_state, next_action, done = batch        # ReBRAC: next_action
        not_done = 1 - done

        with torch.no_grad():
            new_next_action = self.actor_target(next_state)

            # Select action according to actor and add clipped noise
            noise = (torch.randn_like(action) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip
            )

            new_next_action_with_noise = (self.actor_target(next_state) + noise).clamp(
                -self.max_action, self.max_action
            )

            # Compute the target Q value
            target_q1 = self.critic_1_target(next_state, new_next_action_with_noise)
            target_q2 = self.critic_2_target(next_state, new_next_action_with_noise)
            target_q = torch.min(target_q1, target_q2)
            target_q = reward + not_done * (
                    self.gamma * target_q - self.beta2 * F.mse_loss(next_action, new_next_action)
            ) # ReBRAC

        # Get current Q estimates
        current_q1 = self.critic_1(state, action)
        current_q2 = self.critic_2(state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        log_dict["critic_loss"] = critic_loss.item()
        # Optimize the critic
        self.critic_1_optimizer.zero_grad()
        self.critic_2_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.step()

        # Delayed actor updates
        if self.total_it % self.policy_freq == 0:
            # Compute actor loss
            pi = self.actor(state)
            q = self.critic_1(state, pi)

            actor_loss = -q.mean() + self.beta1 * F.mse_loss(pi, action)
            log_dict["actor_loss"] = actor_loss.item()
            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            soft_update(self.critic_1_target, self.critic_1, self.tau)
            soft_update(self.critic_2_target, self.critic_2, self.tau)
            soft_update(self.actor_target, self.actor, self.tau)

        return log_dict

    def state_dict(self) -> Dict[str, Any]:
        return {
            "critic_1": self.critic_1.state_dict(),
            "critic_1_optimizer": self.critic_1_optimizer.state_dict(),
            "critic_2": self.critic_2.state_dict(),
            "critic_2_optimizer": self.critic_2_optimizer.state_dict(),
            "actor": self.actor.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "total_it": self.total_it,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.critic_1.load_state_dict(state_dict["critic_1"])
        self.critic_1_optimizer.load_state_dict(state_dict["critic_1_optimizer"])
        self.critic_1_target = copy.deepcopy(self.critic_1)

        self.critic_2.load_state_dict(state_dict["critic_2"])
        self.critic_2_optimizer.load_state_dict(state_dict["critic_2_optimizer"])
        self.critic_2_target = copy.deepcopy(self.critic_2)

        self.actor.load_state_dict(state_dict["actor"])
        self.actor_optimizer.load_state_dict(state_dict["actor_optimizer"])
        self.actor_target = copy.deepcopy(self.actor)

        self.total_it = state_dict["total_it"]


def main(config: TrainConfig):
    env, eval_env, state_dim, action_dim, replay_buffer, n_episodes, min_return, max_return = preliminary(config)

    max_action = float(env.action_space.high[0])

    actor = Actor(state_dim, action_dim, max_action).to(config.device)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=3e-4)

    critic_1 = Critic(state_dim, action_dim).to(config.device)
    critic_1_optimizer = torch.optim.Adam(critic_1.parameters(), lr=3e-4)
    critic_2 = Critic(state_dim, action_dim).to(config.device)
    critic_2_optimizer = torch.optim.Adam(critic_2.parameters(), lr=3e-4)

    kwargs = {
        "max_action": max_action,
        "actor": actor,
        "actor_optimizer": actor_optimizer,
        "critic_1": critic_1,
        "critic_1_optimizer": critic_1_optimizer,
        "critic_2": critic_2,
        "critic_2_optimizer": critic_2_optimizer,
        "gamma": config.gamma,
        "tau": config.tau,
        "device": config.device,
        # TD3
        "policy_noise": config.policy_noise * max_action,
        "noise_clip": config.noise_clip * max_action,
        "policy_freq": config.policy_freq,
        # ReBRAC
        "beta1": config.beta1,
        "beta2": config.beta2,
    }

    print("---------------------------------------")
    print(f"Training TD3 + BC, Env: {config.env}, Seed: {config.seed}")
    print("---------------------------------------")

    # Initialize actor
    trainer = ReBRAC(**kwargs)

    train_and_eval_loop(trainer, config, replay_buffer, eval_env, actor)


if __name__ == "__main__":
    config = TrainConfig()
    main(config)
