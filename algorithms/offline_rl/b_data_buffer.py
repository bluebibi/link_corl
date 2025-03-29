from typing import List

import torch
import numpy as np

TensorBatch = List[torch.Tensor]

class EpisodeDataBuffer:
    def __init__(
        self,
        n_episodes: int,
        state_dim: int,
        action_dim: int,
        device: str = "cpu",
    ):
        self.n_episodes = n_episodes
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device

        self._states = None
        self._actions = None
        self._rewards = None
        self._next_states = None
        self._next_actions = None
        self._terminations = None
        self._truncations = None
        self._infos = None

        self._return_to_goes = None
        self._episode_lengths = None

    def _to_tensor(self, data: np.ndarray) -> torch.Tensor:
        return torch.tensor(data, dtype=torch.float32, device=self.device)

    def fill_dataset(
            self,
            observations: np.ndarray,
            next_observations: np.ndarray,
            actions: np.ndarray,
            rewards: np.ndarray,
            terminations: np.ndarray,
            truncations: np.ndarray,
            infos: np.ndarray,
            return_to_goes: np.ndarray,
            episode_lengths: np.ndarray
    ):
        self._states = self._to_tensor(observations)
        self._actions = self._to_tensor(actions)
        self._rewards = self._to_tensor(rewards)
        self._next_states = self._to_tensor(next_observations)
        self._terminations = self._to_tensor(terminations)
        self._truncations = self._to_tensor(truncations)
        self._infos = infos

        self._return_to_goes = self._to_tensor(return_to_goes)
        self._episode_lengths = episode_lengths

    def get_all_data_for_dt(self):
        return self._states, self._actions, self._return_to_goes, self._episode_lengths


class DataBuffer:
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

        self._return_to_goes = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self._episode_lengths = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)

        self._infos = None
        self.device = device

    def _to_tensor(self, data: np.ndarray) -> torch.Tensor:
        return torch.tensor(data, dtype=torch.float32, device=self.device)

    # fill data in minari format, i.e. from Dict[str, np.array].
    def _fill_dataset(
            self,
            observations: np.ndarray,
            next_observations: np.ndarray,
            actions: np.ndarray,
            rewards: np.ndarray,
            terminations: np.ndarray,
            truncations: np.ndarray,
            infos: np.ndarray,
            return_to_goes: np.ndarray,
            episode_lengths: np.ndarray
    ):
        if self._size != 0:
            raise ValueError("Trying to load data into non-empty data buffer")
        n_transitions = observations.shape[0]
        if n_transitions > self._buffer_size:
            raise ValueError(
                "Data buffer is smaller than the dataset you are trying to load!"
            )
        self._states[:n_transitions] = self._to_tensor(observations)
        self._actions[:n_transitions] = self._to_tensor(actions)
        self._rewards[:n_transitions] = self._to_tensor(rewards[..., None])
        self._next_states[:n_transitions] = self._to_tensor(next_observations)
        self._terminations[:n_transitions] = self._to_tensor(terminations[..., None])
        self._truncations[:n_transitions] = self._to_tensor(truncations[..., None])
        self._return_to_goes[:n_transitions] = self._to_tensor(return_to_goes[..., None])
        self._episode_lengths[:n_transitions] = self._to_tensor(episode_lengths[..., None])
        self._infos = infos
        self._size += n_transitions
        self._pointer = min(self._size, n_transitions)

        print(f"Dataset size: {n_transitions}")

    def fill_dataset(
            self,
            n_episodes: int,
            observations: np.ndarray,
            next_observations: np.ndarray,
            actions: np.ndarray,
            rewards: np.ndarray,
            terminations: np.ndarray,
            truncations: np.ndarray,
            infos: np.ndarray,
            return_to_goes: np.ndarray,
            episode_lengths: np.ndarray,
            with_next_actions = False
    ):
        n_transitions = observations.shape[0]

        if n_transitions > self._buffer_size:
            raise ValueError(
                "Data buffer is smaller than the dataset you are trying to load!"
            )

        # next_actions
        assert n_transitions % n_episodes == 0

        self._fill_dataset(
            observations, next_observations, actions, rewards, terminations, truncations, infos,
            return_to_goes, episode_lengths
        )

        if with_next_actions:
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
        # Use this method to add new data into the data buffer during fine-tuning.
        # I left it unimplemented since now we do not do fine-tuning.
        raise NotImplementedError