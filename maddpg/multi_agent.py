import numpy as np
import random
import torch

from utils.replaybuffer import ReplayBuffer
from .brain import Brain


class MultiAgent:

    def __init__(self,
                 agent_count,
                 observation_size,
                 action_size,
                 train_config,
                 agent_config,
                 seed=None,
                 actor_model_states=None,
                 critic_model_states=None,
                 device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')):

        def create_brain(idx):
            return Brain(
                agent_count=agent_count,
                observation_size=observation_size,
                action_size=action_size,
                actor_optim_params=train_config['actor_optim_params'],
                critic_optim_params=train_config['critic_optim_params'],
                soft_update_tau=train_config['soft_update_tau'],
                discount_gamma=train_config['discount_gamma'],
                use_batch_norm=False,
                seed=seed,
                actor_network_states=actor_model_states[idx] if actor_model_states else None,
                critic_network_states=critic_model_states[idx] if critic_model_states else None,
                device=device
            )

        self.brains = [create_brain(i) for i in range(agent_count)]
        self.agent_count = agent_count
        self.observation_size = observation_size
        self.action_size = action_size
        self.train_config = train_config
        self.agent_config = agent_config
        self.device = device

        self._batch_size = train_config['mini_batch_size']
        self._update_every = train_config['update_every']

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # Replay memory
        self.memory = ReplayBuffer(
            action_size,
            train_config['buffer_size'],
            self._batch_size,
            device
        )

        self.t_step = 0

    def step(self, obs, actions, rewards, next_obs, dones):
        """observation and learning by replay
        :param obs: array of shape == (agent_count, observation_size)
        :param actions: array of shape == (agent_count, action_size)
        :param rewards: array of shape == (agent_count,)
        :param next_obs: list of  array of shape == (agent_count, observation_size)
        :param dones: array of shape == (agent_count,)
        """
        self.memory.add(obs, actions, rewards, next_obs, dones.astype(np.uint8))

        self.t_step = (self.t_step + 1) % self._update_every

        if self.t_step == 0:
            if len(self.memory) > self._batch_size:
                self._learn()

    def act_torch(self, obs, target, noise=0.0, train=False):
        """Act based on the given batch of observations.
        :param obs: current observation, array of shape == (batch, num_agent, num_stacked_obs*observation_size)
        :param noise: noise factor
        :param train: True for training mode else eval mode
        :return: actions for given state as per current policy.
        """
        actions = [
            brain.act(obs[:, i], target, noise, train)
            for i, brain in enumerate(self.brains)
        ]

        actions = torch.stack(actions).transpose(1, 0)

        return actions

    def act(self, obs, target=False, noise=0.0):
        obs = torch.from_numpy(obs).float().\
            to(self.device).unsqueeze(0)

        with torch.no_grad():
            actions = np.vstack([
                a.cpu().numpy()
                for a in self.act_torch(obs, target, noise)
            ])

        return actions

    def reset(self):
        for brain in self.brains:
            brain.reset()

    def _learn(self):

        experiences = self.memory.sample()
        experiences = self._tensor_experiences(experiences)

        observations, actions, rewards, next_observations, dones = experiences

        all_obs = self._flatten(observations)
        all_actions = self._flatten(actions)
        all_next_obs = self._flatten(next_observations)

        all_target_next_actions = self._flatten(self.act_torch(
            next_observations,
            target=True,
            train=False
        ).contiguous())

        all_local_actions = self.act_torch(
            observations,
            target=False,
            train=True
        ).contiguous()

        for i, brain in enumerate(self.brains):
            # update critics
            brain.update_critic(
                rewards[:, i].unsqueeze(-1), dones[:, i].unsqueeze(-1),
                all_obs, all_actions, all_next_obs, all_target_next_actions
            )

            # update actors
            all_local_actions_agent = all_local_actions.detach()
            all_local_actions_agent[:, i] = all_local_actions[:, i]
            all_local_actions_agent = self._flatten(all_local_actions_agent)
            brain.update_actor(
                all_obs, all_local_actions_agent
            )

            # update targets
            brain.update_targets()

    def _tensor_experiences(self, experiences):
        ob, actions, rewards, next_ob, dones = \
            [torch.from_numpy(e).float().to(self.device) for e in experiences]
        return ob, actions, rewards, next_ob, dones

    @staticmethod
    def _flatten(tensor):
        b, n_agents, d = tensor.shape
        return tensor.view(b, n_agents * d)
