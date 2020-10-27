import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return -lim, lim


class ActorNetwork(nn.Module):

    def __init__(self, observation_size, action_size, use_batch_norm, seed,
                 fc1_units=128, fc2_units=64, fc3_units=32):
        """
        :param observation_size: observation size
        :param action_size: action size
        :param use_batch_norm: True to use batch norm
        :param seed: random seed
        :param fc1_units: number of nodes in 1st hidden layer
        :param fc2_units: number of nodes in 2nd hidden layer
        :param fc3_units: number of nodes in 3rd hidden layer
        """
        super(ActorNetwork, self).__init__()

        if seed is not None:
            torch.manual_seed(seed)

        if use_batch_norm:
            self.bn1 = nn.BatchNorm1d(observation_size)
            self.bn2 = nn.BatchNorm1d(fc1_units)
            self.bn3 = nn.BatchNorm1d(fc2_units)
            self.bn4 = nn.BatchNorm1d(fc3_units)

        # batch norm has bias included, disable linear layer bias
        use_bias = not use_batch_norm

        self.use_batch_norm = use_batch_norm
        self.fc1 = nn.Linear(observation_size, fc1_units, bias=use_bias)
        self.fc2 = nn.Linear(fc1_units, fc2_units, bias=use_bias)
        self.fc3 = nn.Linear(fc2_units, fc3_units, bias=use_bias)
        self.fc4 = nn.Linear(fc3_units, action_size, bias=use_bias)
        self.reset_parameters()

    def forward(self, observation):
        """ map a states to action values
        :param observation: shape == (batch, observation_size)
        :return: action values
        """

        if self.use_batch_norm:
            x = F.relu(self.fc1(self.bn1(observation)))
            x = F.relu(self.fc2(self.bn2(x)))
            x = F.relu(self.fc3(self.bn3(x)))
            return torch.tanh(self.fc4(self.bn4(x)))
        else:
            x = F.relu(self.fc1(observation))
            x = F.relu(self.fc2(x))
            x = F.relu(self.fc3(x))
            return torch.tanh(self.fc4(x))

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(*hidden_init(self.fc3))
        self.fc4.weight.data.uniform_(-3e-3, 3e-3)


class CriticNetwork(nn.Module):

    def __init__(self, observation_size, action_size, use_batch_norm, seed,
                 fc1_units=128, fc2_units=64, fc3_units=32):
        """
        :param observation_size: Dimension of each state
        :param action_size: Dimension of each state
        :param seed: random seed
        :param fc1_units: number of nodes in 1st hidden layer
        :param fc2_units: number of nodes in 2nd hidden layer
        :param fc3_units: number of nodes in 3rd hidden layer
        """
        super(CriticNetwork, self).__init__()

        if seed is not None:
            torch.manual_seed(seed)

        if use_batch_norm:
            self.bn1 = nn.BatchNorm1d(observation_size + action_size)
            self.bn2 = nn.BatchNorm1d(fc1_units)
            self.bn3 = nn.BatchNorm1d(fc2_units)
            self.bn4 = nn.BatchNorm1d(fc3_units)

        # batch norm has bias included, disable linear layer bias
        use_bias = not use_batch_norm

        self.use_batch_norm = use_batch_norm
        self.fc1 = nn.Linear(observation_size + action_size, fc1_units, bias=use_bias)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, fc3_units)
        self.fc4 = nn.Linear(fc3_units, 1)
        self.reset_parameters()

    def forward(self, observation, action):
        """ map (observation, actions) pairs to Q-values
        :param observation: shape == (batch, observation_size)
        :param action: shape == (batch, action_size)
        :return: q-values values
        """

        x = torch.cat([observation, action], dim=1)
        if self.use_batch_norm:
            x = F.relu(self.fc1(self.bn1(x)))
            x = F.relu(self.fc2(self.bn2(x)))
            x = F.relu(self.fc3(self.bn3(x)))
        else:
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = F.relu(self.fc3(x))

        x = self.fc4(x)
        return x

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(*hidden_init(self.fc3))
        self.fc4.weight.data.uniform_(-3e-3, 3e-3)
