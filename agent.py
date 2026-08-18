import numpy as np
from memory_model import memory_model
from collections import deque
from ml_utils import to_single_vector
from random import random, randint

class agent:
    def __init__(self, world_size=5):
        self.model = memory_model()
        self.world_size = world_size

        self.memory_vector = np.zeros([self.model.memory_size, ]).astype(np.float32)
        self.memory = deque(maxlen=10000)
        #each memory will have player_location, target_location, memory, action, reward, next_q_value, terminated
        self.epsilon = 1.
        self.epsilon_decay = 0.999

        self.discount_rate = 0.8

        self.q_value = 0

    def policy(self, observation):
        self.epsilon *= self.epsilon_decay
        self.epsilon = max([0.1, self.epsilon])
        prediction = self.model.forward(np.concat([observation/(self.world_size-1), self.memory_vector/10]))
        action = np.argmax(prediction)
        self.q_value = np.amax(prediction)

        if self.epsilon > random():
            if randint(0, 1) == 0:
                return randint(0, self.model.output_size-1)
            else:
                direction_y = observation[2] - observation[0]
                direction_x = observation[3] - observation[1]
                if abs(direction_x) > abs(direction_y):
                    direction_x = direction_x / abs(direction_x)
                    if direction_x < 0:
                        return 3
                    else:
                        return 2
                else:
                    direction_y = direction_y / abs(direction_y)
                    if direction_y < 0:
                        return 0
                    else:
                        return 1
        else:
            return action

    def update_memory(self):
        self.memory_vector = to_single_vector(self.model.x4)

    def reset_memory(self):
        self.memory_vector *= 0

    def train(self):
        x, y, actions, terminateds = [], [], [], []
        for memory_sample in self.memory:
            player_location, target_location, memory, action, reward, next_q_value, terminated = memory_sample

            x.append(np.concat([player_location/(self.world_size-1), target_location/(self.world_size-1), memory/10]))
            y.append(reward + self.discount_rate * next_q_value)
            actions.append(action)
            terminateds.append(terminated)

        x = np.asarray(x).astype(np.float32)
        y = np.asarray(y).astype(np.float32)

        loss = self.model.backward(x, y, actions)
        return loss