import numpy as np
import yaml
import os
import pickle
from collections import deque
from tensorboardX import SummaryWriter
from environment.Environment import Environment
from maddpg.multi_agent import MultiAgent
from utils.utils import save_agent
from figure import Figure

def train(environment,
          platform_config,
          train_config,
          agent_config,
          print_every,
          mode=None,
          random_seed=None):

    n_episodes = train_config['n_episodes']
    max_t = train_config['max_t']
    ou_noise = train_config['ou_noise_start']
    ou_noise_decay_rate = train_config['ou_noise_decay_rate']
    num_agent = platform_config['n_agent']
    num_task = platform_config['n_task']
    save_every = train_config['save_every']
    task_budget = np.array(platform_config['task_budget_bound'])[0]
    time_budget = np.array(platform_config['time_budget_bound'])[0]
    path_suffix = 'task_budget=' + str(task_budget) + '-time_budget=' + str(time_budget) + \
        '-seed=' + str(random_seed) + '-mode=' + mode + '/'
    result_path = \
        train_config['result_path'] + path_suffix
    model_path = \
        train_config['model_path'] + path_suffix
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    # get the default brain
    env = environment
    env_info = env.reset(train_mode=True)
    writer = SummaryWriter()

    # Initialize our agent
    observation_size = agent_config['obs_size'] * \
        platform_config['n_stacked_observation']
    action_size = agent_config['action_size']
    agent_count = len(env_info.agents)

    multi_agent = MultiAgent(
        agent_count=agent_count,
        observation_size=observation_size,
        action_size=action_size,
        train_config=train_config,
        agent_config=agent_config,
        seed=random_seed
    )

    all_train_welfare = []
    train_welfare_window = deque(maxlen=print_every)

    for i_episode in range(1, n_episodes + 1):

        train_scores = train_episode(env, multi_agent, max_t, ou_noise)
        train_scores = np.min(train_scores)

        ou_noise *= ou_noise_decay_rate
        train_welfare_window.append(train_scores)
        all_train_welfare.append(train_scores)

        for key in env.welfare_episode:
            env.welfare_episode[key].append(mean(env.welfare[key], axis=0))
            env.welfare_episode_window[key].append(env.welfare_episode[key][-1])
            env.welfare_avg[key].append(np.mean(np.array(env.welfare_episode_window[key]), axis=0))

        prices_plot = np.array(env.welfare_avg["prices"])[-1]
        allocation_plot = np.array(env.welfare_avg["allocation"])[-1]
        user_rewards_plot = np.array(env.welfare_avg["MU_rewards"])[-1]
        user_earnings_plot = np.array(env.welfare_avg["MU_earnings"])[-1]
        user_earnings_sum_plot = np.array(env.welfare_avg["MU_earnings_sum"])[-1]
        resource_utilization_plot = np.array(env.welfare_avg["resource_utilization"])[-1]
        ti_earnings_plot = np.array(env.welfare_avg["TI_earnings"])[-1]
        welfare_plot = np.array(env.welfare_avg["welfare"])[-1]
        log_welfare_plot = np.array(env.welfare_avg["log_welfare"])[-1]
        writer.add_scalar('results/MU earnings sum', user_earnings_sum_plot, i_episode)
        writer.add_scalar('results/resource utilization', resource_utilization_plot, i_episode)
        writer.add_scalar('results/welfare', welfare_plot, i_episode)
        writer.add_scalar('results/log welfare', log_welfare_plot, i_episode)
        for i in range(num_agent):
            writer.add_scalars('results/prices', {'MU_{}'.format(i+1): prices_plot[i, 0]}, i_episode)
            writer.add_scalars('results/MU earnings', {'MU_{}'.format(i+1): user_earnings_plot[i]}, i_episode)
            writer.add_scalars('results/MU rewards', {'MU_{}'.format(i+1): user_rewards_plot[i]}, i_episode)
        for j in range(num_task):
            writer.add_scalars('results/TI earnings', {'TI{}'.format(j + 1): ti_earnings_plot[j]}, i_episode)
            for i in range(num_agent):
                writer.add_scalars('results/allocation_TI{}'.format(j+1), {'MU_{}'.format(i+1): allocation_plot[i, j]},
                                   i_episode)

        if i_episode % save_every == 0:
            save_agent(multi_agent, model_path+str(i_episode)+'.pth')
            with open(result_path + str(i_episode) + '.pkl', 'wb') as p:
                pickle.dump(env.welfare_episode, p)
            with open(result_path + str(i_episode) + '_avg.pkl', 'wb') as p:
                pickle.dump(env.welfare_avg, p)

        print('\rEpisode {}\t Minimum rewards: {:.3f}'
              .format(i_episode, np.mean(train_welfare_window)), end='')

        if i_episode % print_every == 0:
            print('\rEpisode {}\tAverage Minimum rewards: {:.3f}'
                  .format(i_episode, np.mean(train_welfare_window)))

    return multi_agent, all_train_welfare


def mean(value, axis=None):

    if not value:
        return 0.0
    elif axis is not None:
        return np.mean(value, axis=axis)
    return np.mean(value)


def train_episode(env, multi_agent, max_t, ou_noise):

    env_info = env.reset(train_mode=True)
    obs = env_info.vector_observations
    multi_agent.reset()
    scores = np.zeros(multi_agent.agent_count)

    for _ in range(max_t):
        actions = multi_agent.act(obs, noise=ou_noise)
        brain_info = env.step(actions)
        next_obs = brain_info.vector_observations
        rewards = np.asarray(brain_info.rewards)
        dones = np.asarray(brain_info.local_done)

        multi_agent.step(obs, actions, rewards, next_obs, dones)
        obs = next_obs
        scores += rewards

        if np.any(dones):
            break

    return scores


if __name__ == '__main__':

    yaml_path = 'environment/platform-config.yaml'

    with open(yaml_path, 'r') as f:
        cfg = yaml.load(f)

    platform_config = cfg['platform_config']
    train_config = cfg['train_config']
    agent_config = cfg['agent_config']
    print_every = cfg['print_every']
    seed = cfg['random_seed']
    task_b = np.array(platform_config['task_budget_bound'])[0]
    time_b = np.array(platform_config['time_budget_bound'])[0]
    result_inf = {
        "prices": [],
        "allocation": [],
        "MU_rewards": [],
        "MU_earnings": [],
        "MU_earnings_sum": [],
        "resource_utilization": [],
        "TI_earnings": [],
        "welfare": [],
        "log_welfare": [],
    }

    env = Environment(agent_config=agent_config, platform_config=platform_config,
                      train_config=train_config, result_inf=result_inf)

    multi_agent, _ = train(
        environment=env,
        platform_config=platform_config,
        train_config=train_config,
        agent_config=agent_config,
        print_every=print_every,
        mode='greedy',
        random_seed=seed
    )

    env.close()

    Figure(task_b, time_b, seed)
    
    

