"Modified buffer from the stable-baselines3 library"
import warnings
from abc import ABC, abstractmethod
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

import numpy as np
import torch as th
from gymnasium import spaces
from mani_skill.trajectory.dataset import ManiSkillTrajectoryDataset

from stable_baselines3.common.preprocessing import get_action_dim, get_obs_shape
from stable_baselines3.common.type_aliases import (
    DictReplayBufferSamples,
    DictRolloutBufferSamples,
    ReplayBufferSamples,
    RolloutBufferSamples,
)
from stable_baselines3.common.utils import get_device
from stable_baselines3.common.vec_env import VecNormalize

try:
    # Check memory used by replay buffer when possible
    import psutil
except ImportError:
    psutil = None

class BaseBuffer(ABC):
    """
    Base class that represent a buffer (rollout or replay)

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device: PyTorch device
        to which the values will be converted
    :param n_envs: Number of parallel environments
    """

    observation_space: spaces.Space
    obs_shape: Tuple[int, ...]

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "auto",
        n_envs: int = 1,
    ):
        super().__init__()
        self.buffer_size = buffer_size
        self.observation_space = observation_space
        self.action_space = action_space
        self.obs_shape = get_obs_shape(observation_space)  # type: ignore[assignment]

        self.action_dim = get_action_dim(action_space)
        self.pos = 0
        self.full = False
        self.device = get_device(device)
        self.n_envs = n_envs
        self.started = False

    @staticmethod
    def swap_and_flatten(arr: np.ndarray) -> np.ndarray:
        """
        Swap and then flatten axes 0 (buffer_size) and 1 (n_envs)
        to convert shape from [n_steps, n_envs, ...] (when ... is the shape of the features)
        to [n_steps * n_envs, ...] (which maintain the order)

        :param arr:
        :return:
        """
        shape = arr.shape
        if len(shape) < 3:
            shape = (*shape, 1)
        return arr.swapaxes(0, 1).reshape(shape[0] * shape[1], *shape[2:])

    def size(self) -> int:
        """
        :return: The current size of the buffer
        """
        if self.full:
            return self.buffer_size
        return self.pos

    def add(self, *args, **kwargs) -> None:
        """
        Add elements to the buffer.
        """
        raise NotImplementedError()

    def extend(self, *args, **kwargs) -> None:
        """
        Add a new batch of transitions to the buffer
        """
        # Do a for loop along the batch axis
        for data in zip(*args):
            self.add(*data)

    def reset(self) -> None:
        """
        Reset the buffer.
        """
        self.pos = 0
        self.full = False

    def sample(self, batch_size: int, env: Optional[VecNormalize] = None):
        """
        :param batch_size: Number of element to sample
        :param env: associated gym VecEnv
            to normalize the observations/rewards when sampling
        :return:
        """
        upper_bound = self.buffer_size if self.full else self.pos
        batch_inds = np.random.randint(0, upper_bound, size=batch_size)
        return self._get_samples(batch_inds, env=env)

    def _get_samples(
        self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None
    ) -> Union[ReplayBufferSamples, RolloutBufferSamples]:
        """
        :param batch_inds:
        :param env:
        :return:
        """
        raise NotImplementedError()

    def to_torch(self, array: np.ndarray, copy: bool = True) -> th.Tensor:
        """
        Convert a numpy array to a PyTorch tensor.
        Note: it copies the data by default

        :param array:
        :param copy: Whether to copy or not the data (may be useful to avoid changing things
            by reference). This argument is inoperative if the device is not the CPU.
        :return:
        """
        if copy:
            return th.tensor(array, device=self.device)
        return th.as_tensor(array, device=self.device)

    @staticmethod
    def _normalize_obs(
        obs: Union[np.ndarray, Dict[str, np.ndarray]],
        env: Optional[VecNormalize] = None,
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        if env is not None:
            return env.normalize_obs(obs)
        return obs

    @staticmethod
    def _normalize_reward(reward: np.ndarray, env: Optional[VecNormalize] = None) -> np.ndarray:
        if env is not None:
            return env.normalize_reward(reward).astype(np.float32)
        return reward


class ReplayBuffer(BaseBuffer):

    observations: np.ndarray
    next_observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray

    demo_observations: np.ndarray
    demo_next_observations: np.ndarray
    demo_actions: np.ndarray
    demo_rewards: np.ndarray
    demo_dones: np.ndarray

    def __init__(
        self,
        args,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "auto",
        n_envs: int = 1,

    ):
        super().__init__(buffer_size, observation_space, action_space, device, n_envs=n_envs)
        self.buffer_size = max(buffer_size // n_envs, 1)
        self.observations = np.zeros((self.buffer_size, self.n_envs, *self.obs_shape), dtype=observation_space.dtype)
        self.next_observations = np.zeros((self.buffer_size, self.n_envs, *self.obs_shape), dtype=observation_space.dtype)
        self.actions = np.zeros((self.buffer_size, self.n_envs, self.action_dim), dtype=action_space.dtype)
        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.args = args
        if args.use_demo:
            self.add_demo_data()

    def add_demo_data(self):
        dataset = ManiSkillTrajectoryDataset(dataset_file=f"demos/{self.args.env_id}/teleop/trajectory.state.pd_joint_delta_pos.h5")
        self.demo_observations = np.array(dataset.data["traj_0"]["obs"]).squeeze(1)[:-1]
        self.demo_next_observations = np.array(dataset.data["traj_0"]["obs"]).squeeze(1)[1:]
        self.demo_actions = np.array(dataset.data["traj_0"]["actions"])
        self.demo_rewards = np.array(dataset.data["traj_0"]["rewards"])
        self.demo_dones =  np.array(dataset.data["traj_0"]["terminated"])
        self.success = np.array(dataset.data["traj_0"]["success"])
        for eps_id in range(1,len(dataset.episodes)):
            eps = dataset.episodes[eps_id]
            trajectory = dataset.data[f"traj_{eps['episode_id']}"]
            self.demo_observations = np.concatenate((self.demo_observations, np.array(trajectory["obs"]).squeeze(1)[:-1]), axis=0)
            self.demo_next_observations = np.concatenate((self.demo_next_observations, np.array(trajectory["obs"]).squeeze(1)[1:]), axis=0)
            self.demo_actions = np.concatenate((self.demo_actions, np.array(trajectory["actions"])), axis=0)
            self.demo_rewards = np.concatenate((self.demo_rewards, np.array(trajectory["rewards"])), axis=0)
            self.demo_dones = np.concatenate((self.demo_dones, np.array(trajectory["terminated"])), axis=0)
            self.success = np.concatenate((self.success, np.array(trajectory["success"])), axis=0)
        print(f"Demo data loaded with {len(self.demo_observations)} samples")
        self.demo_observations = self.demo_observations[self.demo_dones==0]
        self.demo_next_observations = self.demo_next_observations[self.demo_dones==0]
        self.demo_actions = self.demo_actions[self.demo_dones==0]
        if self.args.RESCALE_REWARDS:
            self.demo_rewards[self.success] *=100
        self.demo_rewards = self.demo_rewards[self.demo_dones==0].reshape(-1,1)
        self.demo_dones = self.demo_dones[self.demo_dones==0].reshape(-1,1)
        print(f"Demo data loaded with {len(self.demo_observations)} samples")
        print(self.demo_observations.shape,self.demo_actions.shape,self.demo_next_observations.shape,self.demo_dones.shape,self.demo_rewards.shape)

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:
        action = action.reshape((self.n_envs, self.action_dim))
        self.observations[self.pos] = np.array(obs)
        self.next_observations[self.pos] = np.array(next_obs)
        self.actions[self.pos] = np.array(action)
        self.rewards[self.pos] = np.array(reward)
        self.dones[self.pos] = np.array(done)
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def sample(self, batch_size: int, env: Optional[VecNormalize] = None) -> ReplayBufferSamples:
        if self.args.use_demo:
            demo_bs = int(batch_size*self.args.demo_percent)
            batch_size -= demo_bs
        upper_bound = self.buffer_size if self.full else self.pos
        batch_inds = np.random.randint(0, upper_bound, size=batch_size)
        env_indices = np.random.randint(0, high=self.n_envs, size=(len(batch_inds),))
        next_obs = self._normalize_obs(self.next_observations[batch_inds, env_indices, :],env)
        obs_sample = self._normalize_obs(self.observations[batch_inds, env_indices, :],env)
        ac_sample = self.actions[batch_inds, env_indices, :]
        next_sample = next_obs
        done_sample = self.dones[batch_inds, env_indices].reshape(-1, 1)
        reward_sample = self._normalize_reward(self.rewards[batch_inds, env_indices].reshape(-1, 1),env)
        if self.args.use_demo:
            demo_inds = np.arange(len(self.demo_observations))
            np.random.shuffle(demo_inds)
            suffle_demo_inds = demo_inds[:demo_bs]
            demo_obs_sample = self._normalize_obs(self.demo_observations[suffle_demo_inds, :],env)
            demo_ac_sample = self.demo_actions[suffle_demo_inds, :]
            demo_next_sample = self._normalize_obs(self.demo_next_observations[suffle_demo_inds, :],env)
            demo_done_sample = self.demo_dones[suffle_demo_inds, :].reshape(-1, 1)
            demo_reward_sample = self._normalize_reward(self.demo_rewards[suffle_demo_inds, :], env)
            obs_sample = np.concatenate((demo_obs_sample, obs_sample), axis=0)
            ac_sample = np.concatenate((demo_ac_sample, ac_sample), axis=0)
            next_sample = np.concatenate((demo_next_sample, next_sample), axis=0)
            done_sample = np.concatenate((demo_done_sample, done_sample), axis=0)
            reward_sample = np.concatenate((demo_reward_sample, reward_sample), axis=0)
        data = (
            obs_sample,
            ac_sample,
            next_sample,
            done_sample,
            reward_sample,
        )
        return ReplayBufferSamples(*tuple(map(self.to_torch, data)))


class PrioritizedReplayBuffer(BaseBuffer):

    observations: np.ndarray
    next_observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray

    def __init__(
        self,
        args,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "auto",
        n_envs: int = 1,

    ):
        super().__init__(buffer_size, observation_space, action_space, device, n_envs=n_envs)
        self.buffer_size = max(buffer_size // n_envs, 1)
        self.observations = np.zeros((self.buffer_size, self.n_envs, *self.obs_shape), dtype=observation_space.dtype)
        self.next_observations = np.zeros((self.buffer_size, self.n_envs, *self.obs_shape), dtype=observation_space.dtype)
        self.actions = np.zeros((self.buffer_size, self.n_envs, self.action_dim), dtype=action_space.dtype)
        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.args = args
        self.alpha = self.args.per_alpha
        self.beta_start = self.args.per_beta_start
        self.beta_frames = self.args.per_beta_frames
        self.frame = 1
        self.priorities = np.zeros((self.buffer_size,self.n_envs), dtype=np.float32)

    def beta_by_frame(self, frame_idx):
        return min(1.0, self.beta_start + frame_idx * (1.0 - self.beta_start) / self.beta_frames)
    
    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:
        action = action.reshape((self.n_envs, self.action_dim))
        self.observations[self.pos] = np.array(obs)
        self.next_observations[self.pos] = np.array(next_obs)
        self.actions[self.pos] = np.array(action)
        self.rewards[self.pos] = np.array(reward)
        self.dones[self.pos] = np.array(done)
        if self.started:
            if self.args.use_max_priority:
                max_prio = np.max(self.priorities)
            else:
                max_prio = np.mean(self.priorities) + self.args.sd_scale*np.std(self.priorities)
        else:
            max_prio = 1.0
        self.priorities[self.pos] = np.ones(self.n_envs)*max_prio
        self.started = True
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    #TODO: Implement the update priorities function
    def update_priorities(self, sample_batch_inds,sample_env_indices, prios):
        for b_idx,env_indx,prio in zip(sample_batch_inds,sample_env_indices, prios):
            self.priorities[b_idx,env_indx] = abs(prio) 

    def get_sample_indices(self, batch_size):
        upper_bound = (self.buffer_size if self.full else self.pos)
        priorities = self.priorities.reshape(-1)[:(upper_bound*self.n_envs)]
        probs = priorities ** self.alpha
        probs /= probs.sum()
        index = np.random.choice(np.arange(priorities.size),batch_size, p=probs)
        index_2d = np.unravel_index(index,(upper_bound,self.n_envs))
        beta = self.beta_by_frame(self.frame)
        self.frame+=1
        weights  = (upper_bound*self.n_envs*probs[index]) ** (-beta)
        weights /= weights.max() 
        weights  = np.array(weights, dtype=np.float32) 
        return index_2d[0], index_2d[1],weights

    def sample(self, batch_size: int, env: Optional[VecNormalize] = None) -> ReplayBufferSamples:

        batch_inds, env_indices,weights = self.get_sample_indices(batch_size)
        next_obs = self._normalize_obs(self.next_observations[batch_inds, env_indices, :],env)
        obs_sample = self._normalize_obs(self.observations[batch_inds, env_indices, :],env)
        ac_sample = self.actions[batch_inds, env_indices, :]
        done_sample = self.dones[batch_inds, env_indices].reshape(-1, 1)
        reward_sample = self._normalize_reward(self.rewards[batch_inds, env_indices].reshape(-1, 1), env)
        data = (
            obs_sample,
            ac_sample,
            next_obs,
            done_sample,
            reward_sample,
        )
        return ReplayBufferSamples(*tuple(map(self.to_torch, data))), batch_inds, env_indices, weights