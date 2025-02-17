import minari
import os
import random
from typing import List, Optional, Tuple, Union
import numpy as np
import gymnasium as gym
import torch
import pyrallis
import wandb
import torch.nn as nn
import uuid
from dataclasses import asdict
from pathlib import Path

TensorBatch = List[torch.Tensor]

def compute_mean_std(states: np.ndarray, eps: float) -> Tuple[np.ndarray, np.ndarray]:
    mean = states.mean(0)
    std = states.std(0) + eps
    return mean, std


def normalize_states(states: np.ndarray, mean: np.ndarray, std: np.ndarray):
    return (states - mean) / std

def return_reward_range(all_rewards, all_terminations, max_episode_steps: int) -> Tuple[float, float]:
    returns, lengths = [], []
    ep_ret, ep_len = 0.0, 0
    for r, d in zip(all_rewards, all_terminations):
        ep_ret += float(r)
        ep_len += 1
        if d or ep_len == max_episode_steps:
            returns.append(ep_ret)
            lengths.append(ep_len)
            ep_ret, ep_len = 0.0, 0
    lengths.append(ep_len)  # but still keep track of number of steps
    assert sum(lengths) == len(all_rewards)
    return min(returns), max(returns)

def modify_reward(
    all_rewards,
    all_terminations,
    env_name: str,
    max_episode_steps: int = 1000,
    reward_scale: float = 1.0,
    reward_bias: float = 0.0,
):
    if any(s in env_name for s in ("halfcheetah", "hopper", "walker2d")):
        min_ret, max_ret = return_reward_range(all_rewards, all_terminations, max_episode_steps)
        all_rewards /= max_ret - min_ret
        all_rewards *= max_episode_steps
    all_rewards = all_rewards * reward_scale + reward_bias
    return all_rewards

def wrap_env(
    env: gym.Env,
    state_mean: Union[np.ndarray, float] = 0.0,
    state_std: Union[np.ndarray, float] = 1.0,
    reward_scale: float = 1.0,
) -> gym.Env:
    # PEP 8: E731 do not assign a lambda expression, use a def
    def normalize_state(state):
        return (
            state - state_mean
        ) / state_std  # epsilon should be already added in std.

    def scale_reward(reward):
        # Please be careful, here reward is multiplied by scale!
        return reward_scale * reward

    env = gym.wrappers.TransformObservation(env, normalize_state, observation_space=env.observation_space)
    if reward_scale != 1.0:
        env = gym.wrappers.TransformReward(env, scale_reward)
    return env


class ReplayBuffer:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        buffer_size: int,
        device: str = "cpu",
    ):
        self._buffer_size = buffer_size
        self._pointer = 0
        self._size = 0

        self._states = torch.zeros(
            (buffer_size, state_dim), dtype=torch.float32, device=device
        )
        self._actions = torch.zeros(
            (buffer_size, action_dim), dtype=torch.float32, device=device
        )
        self._rewards = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self._next_states = torch.zeros(
            (buffer_size, state_dim), dtype=torch.float32, device=device
        )
        self._next_actions = torch.zeros(
            (buffer_size, action_dim), dtype=torch.float32, device=device
        )
        self._terminations = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self._truncations = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self._device = device

    def _to_tensor(self, data: np.ndarray) -> torch.Tensor:
        return torch.tensor(data, dtype=torch.float32, device=self._device)

    # Loads data in minari format, i.e. from Dict[str, np.array].
    def load_dataset(
            self,
            observations: np.ndarray,
            next_observations: np.ndarray,
            actions: np.ndarray,
            rewards: np.ndarray,
            terminations: np.ndarray,
            truncations: np.ndarray,
            infos: np.ndarray
    ):
        if self._size != 0:
            raise ValueError("Trying to load data into non-empty replay buffer")
        n_transitions = observations.shape[0]
        if n_transitions > self._buffer_size:
            raise ValueError(
                "Replay buffer is smaller than the dataset you are trying to load!"
            )
        self._states[:n_transitions] = self._to_tensor(observations)
        self._actions[:n_transitions] = self._to_tensor(actions)
        self._rewards[:n_transitions] = self._to_tensor(rewards[..., None])
        self._next_states[:n_transitions] = self._to_tensor(next_observations)
        self._terminations[:n_transitions] = self._to_tensor(terminations[..., None])
        self._truncations[:n_transitions] = self._to_tensor(truncations[..., None])

        self._size += n_transitions
        self._pointer = min(self._size, n_transitions)

        print(f"Dataset size: {n_transitions}")

    def load_dataset_with_next_actions(
            self,
            n_episodes: int,
            observations: np.ndarray,
            next_observations: np.ndarray,
            actions: np.ndarray,
            rewards: np.ndarray,
            terminations: np.ndarray,
            truncations: np.ndarray,
            infos: np.ndarray
    ):
        self.load_dataset(observations, next_observations, actions, rewards, terminations, truncations, infos)

        n_transitions = observations.shape[0]
        if n_transitions > self._buffer_size:
            raise ValueError(
                "Replay buffer is smaller than the dataset you are trying to load!"
            )

        # next_actions
        assert n_transitions % n_episodes == 0
        # actions_unflatten.shape: (1000, 1000, 6)
        actions_unflatten = self._to_tensor(
            actions.reshape(n_episodes, n_transitions // n_episodes, actions.shape[-1])
        )

        # Step 1: 첫 번째(0번째) 항목 삭제 -> Shape (1000, 999, 6)
        next_actions = actions_unflatten[:, 1:, :]

        # Step 2: 마지막 행에 None을 대체할 NaN으로 채운 텐서 생성 -> Shape (1000, 1, 6)
        nan_padding = torch.full((n_episodes, 1, 6), 0.0)  # NaN으로 채우기

        # Step 3: 두 텐서를 concat하여 최종 결과 만들기 -> Shape (1000, 1000, 6)
        next_actions = torch.cat((next_actions, nan_padding), dim=1)

        # next_actions_flatten.shape: (1000000, 6)
        next_actions_flatten = next_actions.reshape(-1, actions.shape[-1])

        self._next_actions[:n_transitions] = next_actions_flatten

    def sample(self, batch_size: int) -> TensorBatch:
        indices = np.random.randint(0, min(self._size, self._pointer), size=batch_size)
        states = self._states[indices]
        actions = self._actions[indices]
        rewards = self._rewards[indices]
        next_states = self._next_states[indices]
        dones = torch.where((self._terminations[indices] + self._truncations[indices]) > 0.0, 1.0, 0.0)
        return [states, actions, rewards, next_states, dones]

    def sample_with_next_actions(self, batch_size: int) -> TensorBatch:
        indices = np.random.randint(0, min(self._size, self._pointer), size=batch_size)
        states = self._states[indices]
        actions = self._actions[indices]
        rewards = self._rewards[indices]
        next_states = self._next_states[indices]
        next_actions = self._next_actions[indices]
        dones = torch.where((self._terminations[indices] + self._truncations[indices]) > 0.0, 1.0, 0.0)
        # print(self._terminations[indices].sum(), self._truncations[indices].sum(), dones.shape, dones.sum(), "@#############")
        return [states, actions, rewards, next_states, next_actions, dones]

    def get_all_states_and_actions(self):
        return self._states[:self._size], self._actions[:self._size]

    def add_transition(self):
        # Use this method to add new data into the replay buffer during fine-tuning.
        # I left it unimplemented since now we do not do fine-tuning.
        raise NotImplementedError

def set_seed(
    seed: int, env: Optional[gym.Env] = None, deterministic_torch: bool = False
):
    if env is not None:
        env.observation_space.seed(seed)
        env.action_space.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(deterministic_torch)

def soft_update(target: nn.Module, source: nn.Module, tau: float):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_((1 - tau) * target_param.data + tau * source_param.data)

def wandb_init(config: dict) -> None:
    wandb.init(
        config=config,
        project=config["project"],
        group=config["group"],
        name=config["name"],
        id=str(uuid.uuid4()),
    )
    wandb.run.save()

def preliminary(config):
    dataset = minari.load_dataset(config.minari_dataset_name, download=True)

    n_episodes = dataset.total_episodes

    env = dataset.recover_environment()
    env.observation_space.seed(config.seed)
    env.action_space.seed(config.seed)

    eval_env = dataset.recover_environment(eval_env=True)
    assert env.spec == eval_env.spec

    # Set seeds
    seed = config.seed
    set_seed(seed, env)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    print("state_dim: ", state_dim)
    print("action_dim: ", action_dim)
    print("state_space: ", env.observation_space)
    print("action_space: ", env.action_space)

    all_episode_list = dataset.sample_episodes(n_episodes=n_episodes)

    # (1000, 1001, 17)
    all_observations = np.asarray([episode.observations for episode in all_episode_list])

    # (1000, 1000, 6)
    all_actions = np.asarray([episode.actions for episode in all_episode_list])

    # (1000, 1000)
    all_rewards = np.asarray([episode.rewards for episode in all_episode_list])

    # (1000, 1000)
    all_terminations = np.asarray([episode.terminations for episode in all_episode_list])

    # (1000, 1000)
    all_truncations = np.asarray([episode.truncations for episode in all_episode_list])
    # print(all_truncations.sum(axis=-1).sum(), "@@@@@")

    # (1000,): [{} {} ... {} {}]
    all_infos = np.asarray([episode.infos for episode in all_episode_list])

    # print("all_observations.shape:", all_observations.shape)
    # print("all_actions.shape:", all_actions.shape)
    # print("all_rewards.shape:", all_rewards.shape)
    # print("all_terminations.shape:", all_terminations.shape)
    # print("all_truncations.shape:", all_truncations.shape)
    # print("all_infos.shape:", all_infos.shape)

    # Print Episode Rewards
    episode_rewards = all_rewards.sum(axis=1)
    average_episode_rewards = episode_rewards.mean().item()
    print(f"Average Episode Reward over {n_episodes} episodes: {average_episode_rewards:.2f}")

    if config.normalize_reward:
        all_rewards = modify_reward(
            all_rewards.reshape(-1),
            np.where((all_terminations + all_truncations) > 0.0, 1.0, 0.0).reshape(-1),
            config.env,
            reward_scale=config.reward_scale,
            reward_bias=config.reward_bias,
        ).reshape(n_episodes, -1)

    if config.normalize:
        state_mean, state_std = compute_mean_std(all_observations.reshape(-1, state_dim), eps=1e-3)
    else:
        state_mean, state_std = 0, 1

    # (1000 * 1001, 17)
    all_observations = normalize_states(
        all_observations.reshape(-1, state_dim), state_mean, state_std
    )

    # (1000 * 1001, 17) -> (1000, 1001, 17)
    all_observations = all_observations.reshape(n_episodes, -1, state_dim)

    # (1000, 1000, 17)
    observations = all_observations[:, :-1, :]

    # (1000, 1000, 17)
    next_observations = all_observations[:, 1:, :]

    env = wrap_env(env, state_mean=state_mean, state_std=state_std)
    eval_env = wrap_env(eval_env, state_mean=state_mean, state_std=state_std)

    replay_buffer = ReplayBuffer(
        state_dim,
        action_dim,
        config.buffer_size,
        config.device,
    )

    if config.name.startswith("ReBRAC"):
        replay_buffer.load_dataset_with_next_actions(
            n_episodes,
            observations.reshape(-1, state_dim),
            next_observations.reshape(-1, state_dim),
            all_actions.reshape(-1, action_dim),
            all_rewards.reshape(-1),
            all_terminations.reshape(-1),
            all_truncations.reshape(-1),
            all_infos.reshape(-1)
        )
    else:
        replay_buffer.load_dataset(
            observations.reshape(-1, state_dim),
            next_observations.reshape(-1, state_dim),
            all_actions.reshape(-1, action_dim),
            all_rewards.reshape(-1),
            all_terminations.reshape(-1),
            all_truncations.reshape(-1),
            all_infos.reshape(-1)
        )

    # max_action = float(env.action_space.high[0])

    if config.checkpoints_path is not None:
        print(f"Checkpoints path: {config.checkpoints_path}")
        os.makedirs(config.checkpoints_path, exist_ok=True)
        with open(os.path.join(config.checkpoints_path, "config.yaml"), "w") as f:
            pyrallis.dump(config, f)

    return env, eval_env, state_dim, action_dim, replay_buffer

@torch.no_grad()
def eval_actor(
    env: gym.Env, actor: nn.Module, device: str, n_episodes: int, seed: int
):
    actor.eval()
    episode_rewards = []
    episode_timesteps = []

    for _ in range(n_episodes):
        state, done = env.reset(), False
        state = state[0]

        episode_reward = 0.0
        timestep = 0

        while not done:
            action = actor.act(state, device)
            state, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            timestep += 1
            done = terminated or truncated
        episode_rewards.append(episode_reward)
        episode_timesteps.append(timestep)

    actor.train()

    return np.asarray(episode_rewards), np.asarray(episode_timesteps)

def get_gamma(t, config):
    return config.gamma_start + (config.gamma_end - config.gamma_start) * (t / config.max_timesteps)

def train_and_eval_loop(trainer, config, replay_buffer, eval_env, actor):
    if config.load_model != "":
        policy_file = Path(config.load_model)
        trainer.load_state_dict(torch.load(policy_file))
        actor = trainer.actor

    if config.wandb:
        wandb_init(asdict(config))

    evaluations = []
    for t in range(int(config.max_timesteps)):
        if config.name.startswith("ReBRAC"):
            batch = replay_buffer.sample_with_next_actions(config.batch_size)
        else:
            batch = replay_buffer.sample(config.batch_size)

        batch = [b.to(config.device) for b in batch]

        if config.name.startswith("ReBRAC"):
            config.gamma = get_gamma(t, config)

        ########### TRAIN #############
        log_dict = trainer.train(batch)
        ###############################

        if config.wandb:
            wandb.log(log_dict, step=trainer.total_it)

        # Evaluate episode
        if (t + 1) % config.eval_freq == 0:
            print(f"Time steps: {t + 1}")
            eval_scores, eval_timesteps = eval_actor(
                eval_env,
                actor,
                device=config.device,
                n_episodes=config.n_episodes,
                seed=config.seed,
            )
            eval_score = eval_scores.mean()
            eval_timestep = eval_timesteps.mean()

            # normalized_eval_score = eval_env.get_normalized_score(eval_score) * 100.0
            evaluations.append(eval_score)
            print("---------------------------------------")
            print(
                f"Evaluation episode reward over {config.n_episodes} episodes: {eval_score:.3f}"
            )
            print(
                f"Evaluation timestep over {config.n_episodes} episodes: {eval_timestep:.2f}"
            )
            print("---------------------------------------")
            if config.checkpoints_path:
                torch.save(
                    trainer.state_dict(),
                    os.path.join(config.checkpoints_path, f"checkpoint_{t}.pt"),
                )
            if config.wandb:
                wandb.log(
                    {
                        "eval_episode_reward": eval_score,
                        "eval_timestep": eval_timestep
                    }, step=trainer.total_it,
                )