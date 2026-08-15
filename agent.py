import numpy as np
from memory_model import memory_model
from environment import gridworld
from collections import deque
from ml_utils import to_single_vector
from random import random, randint

class agent:
    def __init__(self):
        self.model = memory_model()

        self.memory_vector = np.zeros([self.model.memory_size, ]).astype(np.float32)
        self.memory = deque(maxlen=100000)
        self.epsilon = 1.
        self.epsilon_decay = 0.999

    def policy(self, observation):
        self.epsilon *= self.epsilon_decay
        self.epsilon = max([0.1, self.epsilon])
        action = np.argmax(self.model.forward(np.concat([observation, self.memory])))
        self.memory_vector = to_single_vector(self.model.x4)
        if self.epsilon > random():            
            return randint(0, self.model.memory_size-1)
        else:
            return action

