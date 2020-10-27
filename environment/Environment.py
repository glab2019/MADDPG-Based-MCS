import numpy as np
from .Brain import BrainInfo, Platform
import cvxpy as cvx
from collections import deque
import torch

epsilon = 1e-2
mu = 0.05

class Environment(object):
    """
    params: n_agent: the number of mobile users;
    params: all_agent_params: mobile users' params consisting of cost ci, whether to participate mi, reward ri;
    params: all_environment_params: environment's params consisting of n_agents, n_tasks, weights Wij,
            strategy xij, task_budget bj, time_budget ti, reward r_p, price_bound;
    """
    def __init__(self, agent_config=None, platform_config=None, train_config=None, result_inf=None):
        self.platform = Platform(config=platform_config)
        self._n_agent = self.platform.n_agent
        self._n_task = self.platform.n_task
        self.n_stacked_observation = self.platform.n_stacked_observation
        self.brain_info = BrainInfo(agents=np.array([i for i in range(self._n_agent)]))
        self.price_bound = self.platform.price_bound
        self.action_size = agent_config['action_size']
        self.obs_size = agent_config['obs_size']
        self.len_window = train_config['len_window']
        self.penalty_factor = platform_config['penalty_factor']
        self.alloc_thr = platform_config['allocation_threshold']
        self.welfare = dict(result_inf)
        self.welfare_episode = dict(result_inf)
        self.welfare_avg = dict(result_inf)
        self.welfare_episode_window = dict(result_inf)
        for key in result_inf:
            self.welfare_episode_window[key] = deque(maxlen=self.len_window)

    def get_num_agents(self):
        return self._n_agent

    def get_action_size(self):
        return self.action_size

    def reset(self, train_mode=True) -> BrainInfo:

        """
        Sends a signal to reset the environment.
        :return: AllBrainInfo: A Data structure corresponded to the initial set state of the environment.
        """
        # TODO train_mode
        self.platform.reset()
        self.brain_info.rewards = np.zeros(self._n_agent)
        self.brain_info.feedback = np.zeros(self._n_agent)
        self.brain_info.vector_observations = np.zeros((self._n_agent, self.n_stacked_observation*self.obs_size))
        self.brain_info.local_done = [False for i in range(self._n_agent)]
        self.brain_info.previous_earning = np.ones((1, self._n_agent)) * 0
        for key in self.welfare:
            self.welfare[key] = []
        actions = np.random.random((self._n_agent, self.action_size))
        self.step(actions)
        return self.brain_info

    def step(self, vector_action=None) -> BrainInfo:
        """
        Provides the environment with an action, moves the environment dynamics forward forward accordingly,
        and returns observation, state, reward information to the agent.
        :param vector_action: Agent's vector action to send to environment. Can be a scalar or vector of int/floats.
        :return: AllBrainInfo: A Data structure corresponding to the new state of the environment.
        """
        platform = self.platform
        s = self.process(vector_action=vector_action, platform=platform)
        return s

    def process(self, vector_action, platform):
        """
        input actions(n_agent * action_size),the system return the observation, rewards and local dones,
        :param vector_action:array of shape == (agent_count * action_size) =2(MU) * 1(the price of goods)
        :param platform:
        :return:
        """
        bound = platform.price_bound
        time_budget = platform.time_budget
        task_budget = platform.task_budget
        weights = platform.weights
        price = np.maximum(np.array(vector_action), 0) * bound
        task_allocation, log_welfare = self.platform_opti(price=price, time_budget=time_budget,
                                                          task_budget=task_budget, weights=weights)
        allocation = np.sum(task_allocation, axis=1)
        ti_earnings = np.sum(weights*task_allocation-price*task_allocation, axis=0)
        welfare = np.sum(weights*task_allocation)
        self.brain_info.vector_observations[:, :(self.n_stacked_observation-1)*self.obs_size] = \
            self.brain_info.vector_observations[:, self.obs_size:]
        self.brain_info.vector_observations[:, -2] = price[:, 0]
        self.brain_info.vector_observations[:, -1] = allocation[:]
        allocation_proportion = allocation/np.sum(time_budget)
        mu_earning = np.squeeze(price)*allocation
        mu_earning_sum = np.sum(mu_earning)
        # rewards = np.log(np.maximum(mu_earning+epsilon+1-mu*allocation, epsilon))/10
        rewards = np.log(np.maximum(mu_earning+1-mu*allocation, epsilon)) / 10
        self.brain_info.rewards = rewards
        self.brain_info.previous_earning = np.array(mu_earning)
        self.brain_info.previous_action = np.array(vector_action)
        self.welfare["prices"].append(price)
        self.welfare["allocation"].append(task_allocation)
        self.welfare["MU_rewards"].append(rewards)
        self.welfare["MU_earnings"].append(mu_earning)
        self.welfare["MU_earnings_sum"].append(mu_earning_sum)
        self.welfare["resource_utilization"].append(np.sum(allocation_proportion))
        self.welfare["TI_earnings"].append(ti_earnings)
        self.welfare["welfare"].append(welfare)
        self.welfare["log_welfare"].append(log_welfare)
        return self.brain_info

    def platform_opti(self, price, time_budget, task_budget, weights):
        """
        compute tasks allocation for agents, platform_rewards,
        :param price: agents price for tasks: shape(n_agents,)
        :param time_budget: time budget of each mobile user: shape(n_agent,)
        :param task_budget: budget of task initiator: shape(n_task)
        :param weights: value weights of MUs contribute to tasks
        :return:allocation_agent: shape(n_agent,) and rewards(scalar,)
        """
        x = cvx.Variable((self._n_agent, self._n_task))
        w = np.ones((self._n_task, 1))
        objective = cvx.Maximize(cvx.sum(cvx.log(1+cvx.sum(cvx.multiply(x, weights), axis=0))))
        constraints = [x >= 0, cvx.matmul(x, w) <= time_budget, cvx.matmul(x.T, price) <= task_budget]
        prob = cvx.Problem(objective, constraints)
        rewards = prob.solve(solver=cvx.SCS)
        return x.value, rewards

    def close(self):
        self.reset()
