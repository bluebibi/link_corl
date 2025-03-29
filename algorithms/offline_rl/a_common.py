import minari
import os
import random
from typing import List, Tuple, Union
import numpy as np
import gymnasium as gym
import torch
import pyrallis
import wandb
import torch.nn as nn
import uuid
from dataclasses import asdict
from pathlib import Path

from algorithms.offline_rl.b_data_buffer import DataBuffer, EpisodeDataBuffer


def compute_mean_std(states: np.ndarray, eps: float) -> Tuple[np.ndarray, np.ndarray]:
    mean = states.mean(0)
    std = states.std(0) + eps
    return mean, std


def normalize_states(states: np.ndarray, mean: np.ndarray, std: np.ndarray):
    return (states - mean) / std

def return_reward_range(all_rewards, all_terminations, max_episode_steps: int) -> Tuple[float, float]:
    returns, lengths = [], []

    episode_return, episode_length = 0.0, 0
    for r, d in zip(all_rewards, all_terminations):
        episode_return += float(r)
        episode_length += 1
        if d or episode_length == max_episode_steps:
            returns.append(episode_return)
            lengths.append(episode_length)
            episode_return, episode_len = 0.0, 0

    lengths.append(episode_length)  # but still keep track of number of steps

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
    min_return = None
    max_return = None

    if any(s in env_name for s in ("halfcheetah", "hopper", "walker2d")):
        min_return, max_return = return_reward_range(all_rewards, all_terminations, max_episode_steps)
        all_rewards /= max_return - min_return
        all_rewards *= max_episode_steps

    all_rewards = all_rewards * reward_scale + reward_bias

    return all_rewards, min_return, max_return

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


def set_seed(
    seed: int, deterministic_torch: bool = False
):
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

def pad_along_axis(
    arr: np.ndarray, pad_to: int, axis: int = 0, fill_value: float = 0.0, device = None
):
    pad_size = pad_to - arr.shape[axis]
    if pad_size <= 0:
        return arr

    npad = [(0, 0)] * arr.ndim
    npad[axis] = (0, pad_size)

    return torch.tensor(
        np.pad(arr, pad_width=npad, mode="constant", constant_values=fill_value), dtype=torch.float32, device=device
    )

def discounted_cumulative_sum(x: np.ndarray, gamma: float) -> np.ndarray:
    cumulative_sum = np.zeros_like(x)
    cumulative_sum[-1] = x[-1]
    for t in reversed(range(x.shape[0] - 1)):
        cumulative_sum[t] = x[t] + gamma * cumulative_sum[t + 1]
    return cumulative_sum

def preliminary(config):
    dataset = minari.load_dataset(config.minari_dataset_name, download=True)
    n_episodes = dataset.total_episodes
    min_return = None
    max_return = None

    env = dataset.recover_environment()
    env.observation_space.seed(config.train_seed)
    env.action_space.seed(config.train_seed)

    eval_env = dataset.recover_environment(eval_env=True)
    eval_env.observation_space.seed(config.eval_seed)
    eval_env.action_space.seed(config.eval_seed)

    assert env.spec == eval_env.spec

    # Set seeds
    set_seed(config.seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    print("state_dim: ", state_dim)
    print("action_dim: ", action_dim)
    print("state_space: ", env.observation_space)
    print("action_space: ", env.action_space)

    all_episode_list = dataset.iterate_episodes()

    all_observations = []
    all_actions = []
    all_rewards = []
    all_terminations = []
    all_truncations = []
    all_infos = []
    all_return_to_gos = []
    all_episode_lengths = []

    for episode in all_episode_list:
        all_observations.append(episode.observations)
        all_actions.append(episode.actions)
        all_rewards.append(episode.rewards)
        all_terminations.append(episode.terminations)
        all_truncations.append(episode.truncations)
        all_infos.append(episode.infos)
        all_return_to_gos.append(discounted_cumulative_sum(episode.rewards, gamma=1.0))
        all_episode_lengths.append(len(episode.actions))

    all_observations = np.asarray(all_observations)
    all_actions = np.asarray(all_actions)
    all_rewards = np.asarray(all_rewards)
    all_terminations = np.asarray(all_terminations)
    all_truncations = np.asarray(all_truncations)
    all_infos = np.asarray(all_infos)
    all_return_to_goes = np.asarray(all_return_to_gos)
    all_episode_lengths = np.asarray(all_episode_lengths)

    print("#" * 50)
    print("all_observations.shape:", all_observations.shape)
    print("all_actions.shape:", all_actions.shape)
    print("all_rewards.shape:", all_rewards.shape)
    print("all_terminations.shape:", all_terminations.shape)
    print("all_truncations.shape:", all_truncations.shape)
    print("all_infos.shape:", all_infos.shape)
    print("all_return_to_goes.shape:", all_return_to_goes.shape)
    print("all_episode_lengths.shape:", all_episode_lengths.shape)
    print("#" * 50)

    # Print Episode Rewards
    episode_rewards = np.sum(all_rewards, axis=1)
    average_episode_rewards = np.mean(episode_rewards)
    print(f"Average Episode Reward over {n_episodes} episodes: {average_episode_rewards:.2f}")

    if config.normalize_reward:
        all_rewards, min_return, max_return = modify_reward(
            all_rewards.reshape(-1),
            np.where((all_terminations + all_truncations) > 0.0, 1.0, 0.0).reshape(-1),
            config.env,
            reward_scale=config.reward_scale,
            reward_bias=config.reward_bias,
        )

        all_rewards = all_rewards.reshape(n_episodes, -1)

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

    if config.name.startswith("DT"):
        data_buffer = EpisodeDataBuffer(
            n_episodes,
            state_dim,
            action_dim,
            config.device,
        )
        data_buffer.fill_dataset(
            observations,
            next_observations,
            all_actions,
            all_rewards,
            all_terminations,
            all_truncations,
            all_infos,
            all_return_to_goes,
            all_episode_lengths
        )
    else:
        data_buffer = DataBuffer(
            state_dim,
            action_dim,
            config.buffer_size,
            config.device,
        )
        data_buffer.fill_dataset(
            n_episodes,
            observations.reshape(-1, state_dim),
            next_observations.reshape(-1, state_dim),
            all_actions.reshape(-1, action_dim),
            all_rewards.reshape(-1),
            all_terminations.reshape(-1),
            all_truncations.reshape(-1),
            all_infos.reshape(-1),
            all_return_to_goes.reshape(-1),
            all_episode_lengths.reshape(-1),
            with_next_actions = True if config.name.startswith("ReBRAC") else False
        )

    # max_action = float(env.action_space.high[0])

    if config.checkpoints_path is not None:
        print(f"Checkpoints path: {config.checkpoints_path}")
        os.makedirs(config.checkpoints_path, exist_ok=True)
        with open(os.path.join(config.checkpoints_path, "config.yaml"), "w") as f:
            pyrallis.dump(config, f)

    return env, eval_env, state_dim, action_dim, data_buffer, n_episodes, min_return, max_return

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

def train_and_eval_loop(trainer, config, data_buffer, eval_env, actor):
    if config.load_model != "":
        policy_file = Path(config.load_model)
        trainer.load_state_dict(torch.load(policy_file))
        actor = trainer.actor

    if config.wandb:
        wandb_init(asdict(config))

    evaluations = []
    for t in range(int(config.max_timesteps)):
        if config.name.startswith("ReBRAC"):
            batch = data_buffer.sample_with_next_actions(config.batch_size)
        else:
            batch = data_buffer.sample(config.batch_size)

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