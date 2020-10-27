import torch
import os

from maddpg.multi_agent import MultiAgent


def save_agent(multi_agent, path):
    """ Save agent to path
    :param agent: `MultiAgent` object
    :param path: target file path
    """

    actor_model_states = \
        [b.get_actor_model_states() for b in multi_agent.brains]

    critic_model_states = \
        [b.get_critic_model_states() for b in multi_agent.brains]

    model_dict = {
        'agent_count': multi_agent.agent_count,
        'observation_size': multi_agent.observation_size,
        'action_size': multi_agent.action_size,
        'train_config': multi_agent.train_config,
        'agent_config': multi_agent.agent_config,
        'actor_models': actor_model_states,
        'critic_models': critic_model_states
    }

    path_dir, _ = os.path.split(path)

    if not os.path.exists(path_dir):
        os.makedirs(path_dir)

    torch.save(model_dict, path)

    print('\nmodel is saved to %s' % path)


def load_agent(path):
    """ Load agent from path
    :param path: source file path
    :return: `Agent` object
    """

    model_dict = torch.load(
        path,
        map_location=lambda storage, loc: storage
    )

    multi_agent = MultiAgent(
        agent_count=model_dict['agent_count'],
        observation_size=model_dict['observation_size'],
        action_size=model_dict['action_size'],
        train_config=model_dict['train_config'],
        agent_config=model_dict['agent_config'],
        actor_model_states=model_dict['actor_models'],
        critic_model_states=model_dict['critic_models']
    )

    print('\nmodel is loaded from %s' % path)
    print('  - agent count %s' % multi_agent.agent_count)
    print('  - observation size %s' % multi_agent.observation_size)
    print('  - action size %s' % multi_agent.action_size)
    print('  - train config %s' % multi_agent.train_config)
    print('  - agent config %s' % multi_agent.agent_config)

    return multi_agent
