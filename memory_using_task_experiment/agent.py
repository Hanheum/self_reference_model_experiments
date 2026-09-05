import numpy as np
from memory_model import memory_model
from collections import deque
from ml_utils import to_single_vector
from random import random, randint

class agent:
    def __init__(self, world_size=5):
        self.model = memory_model()
        self.target_model = memory_model()

        self.update_target_model()

        self.world_size = world_size

        self.memory_vector = np.zeros([self.model.memory_size, ]).astype(np.float32)
        self.memory = deque(maxlen=100000)
        #each memory will have player_location, target_location, target_type, next_player_location, next_target_location, next_target_type, memory, next_memory, action, reward, terminated
        self.epsilon = 1.
        self.epsilon_decay = 0.999999

        self.q_value = 0

        self.discount_rate = 0.8
        self.batch_size = 64

    def policy(self, observation, eval=False):
        self.epsilon *= self.epsilon_decay
        self.epsilon = max([0.1*float(not eval), self.epsilon])
        prediction = self.model.forward(np.concat([observation/(self.world_size-1), self.memory_vector/10]))
        action = np.argmax(prediction)
        self.q_value = np.amax(prediction)

        if self.epsilon > random():
            return randint(0, self.model.output_size-1)
        else:
            return action

    def update_memory(self):
        self.memory_vector = to_single_vector(self.model.x4)

    def reset_memory(self):
        self.memory_vector *= 0

    def update_target_model(self):
        self.target_model.import_variables(*self.model.export_variables())

    def train(self):
        batch_idx = np.random.choice(list(range(len(self.memory))), self.batch_size, replace=False)
        batch = []
        for i in batch_idx:
            batch.append(self.memory[i])

        x, y, actions, rewards, terminateds = [], [], [], [], []
        for memory_sample in batch:
            player_location, target_location, target_type, next_player_location, next_target_location, next_target_type, memory, next_memory, action, reward, terminated = memory_sample

            x.append(np.concat([player_location/(self.world_size-1), target_location/(self.world_size-1), target_type, memory/10]))
            y.append(np.concat([next_player_location/(self.world_size-1), next_target_location/(self.world_size-1), next_target_type, next_memory/10]))
            actions.append(action)
            rewards.append(reward)
            terminateds.append(terminated)

        x = np.asarray(x).astype(np.float32)
        y = np.asarray(y).astype(np.float32)

        rewards = np.asarray(rewards).astype(np.float32)
        terminateds = np.asarray(terminateds).astype(np.float32)

        y = np.amax(self.target_model.forward_train(y), axis=1)
        y = rewards + self.discount_rate * y * (1 - terminateds)

        loss = self.model.backward(x, y, actions)
        return loss