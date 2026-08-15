import numpy as np
from memory_model import memory_model
from environment import gridworld
from collections import deque
from ml_utils import to_single_vector
from random import random, randint

class agent:
    def __init__(self, world_size=5):
        self.model = memory_model()
        self.world_size = world_size

        self.memory_vector = np.zeros([self.model.memory_size, ]).astype(np.float32)
        self.memory = deque(maxlen=100000)
        #each memory will have player_location, target_location, memory, action, reward, terminated
        self.epsilon = 1.
        self.epsilon_decay = 0.999

        self.discount_rate = 0.5

    def policy(self, observation):
        self.epsilon *= self.epsilon_decay
        self.epsilon = max([0.1, self.epsilon])
        action = np.argmax(self.model.forward(np.concat([observation, self.memory_vector])))

        if self.epsilon > random():            
            return randint(0, self.model.output_size-1)
        else:
            return action

    def update_memory(self):
        self.memory_vector = to_single_vector(self.model.x4)

    def reset_memory(self):
        self.memory_vector *= 0

    def train(self):
        x, rewards, actions, terminateds = [], [], [], []
        for memory_sample in self.memory:
            player_location, target_location, memory, action, reward, terminated = memory_sample
            x.append(np.concat([player_location/(self.world_size-1), target_location/(self.world_size-1), memory]))
            actions.append(action)
            rewards.append(reward)
            terminateds.append(terminated)

        x = np.asarray(x).astype(np.float32)

        summed_chunks = []

        startpoint, endpoint = 0, 0
        for i, terminated in enumerate(terminateds):
            if terminated:
                endpoint = i
                reward_chunk = np.array(rewards[startpoint:endpoint+1])
                summed_chunk = np.zeros_like(reward_chunk)
                for j in range(len(reward_chunk)):
                    k = len(reward_chunk) - j
                    summed_chunk[0:k] *= self.discount_rate
                    summed_chunk[0:k] += reward_chunk[k-1]

                startpoint = i + 1 
                summed_chunks.append(summed_chunk)

        summed_rewards = np.concat(summed_chunks)

#code below here are test codes for Agent class. will be deleted after some verifications.

world = gridworld()
Agent = agent()

for a in range(2):
    terminated = False
    player_location, target_location = world.reset()
    Agent.reset_memory()
    while not terminated:
        action = Agent.policy(np.concat([player_location, target_location]))
        new_player_location, new_target_location, reward, terminated = world.step(action)
        Agent.memory.append([player_location, target_location, Agent.memory_vector, action, reward, terminated])
        Agent.update_memory()

        player_location = new_player_location
        target_location = new_target_location

Agent.train()
print(len(Agent.memory))