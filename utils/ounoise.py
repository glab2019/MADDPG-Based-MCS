import numpy as np
import random
import copy


class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.state = None
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state

        # W => wiener process with ~N(0, 1)
        dW = np.random.randn(*x.shape)

        dx = self.theta * (self.mu - x) + self.sigma * dW

        self.state = x + dx
        return self.state
