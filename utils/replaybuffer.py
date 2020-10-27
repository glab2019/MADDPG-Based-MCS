from collections import namedtuple, deque
import numpy as np
import random


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, device):
        """Initialize a ReplayBuffer object.
        :param action_size: dimension of each action
        :param buffer_size: maximum size of buffer
        :param batch_size: size of each training batch
        :param device: device for tensor construction
        """
        self.action_size = action_size

        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size

        self.experience = namedtuple(
            "Experience",
            field_names=[
                "observations",
                "actions",
                "rewards",
                "next_observations",
                "dones",
            ]
        )

        self.device = device

    def add(self, obs, actions, rewards, next_obs, dones):
        """Add a new experience to memory.
        :param obs: shape == (n_agents, observation_size)
        :param actions: shape == (n_agents, action_size)
        :param rewards: shape == (n_agents,)
        :param next_obs: shape == (n_agents, observatio_size)
        :param dones: shape == (n_agents,)
        """
        exp = self.experience(obs, actions, rewards, next_obs, dones)
        self.memory.append(exp)

    def sample(self):
        """Sample a batch of experiences from memory."""

        experiences = random.sample(self.memory, k=self.batch_size)

        # transpose and convert to numpy array
        observations, actions, rewards, next_observations, dones = \
            list(map(lambda x: np.asarray(x), zip(*experiences)))

        return observations, actions, rewards, next_observations, dones

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
