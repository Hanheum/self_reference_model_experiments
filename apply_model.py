import numpy as np
from environment import gridworld
from agent import agent
from time import sleep

Agent = agent()
world = gridworld()

Agent.model.load('./model_weight')
Agent.epsilon = 0.3

count_reached_targets = 0

for i in range(10):
    terminated = False
    player_location, target_location = world.reset()
    total_reward = 0

    while not terminated:
        action = Agent.policy(np.concat([player_location, target_location]))
        player_location, target_location, reward, terminated = world.step(action)
        Agent.update_memory()
        total_reward += reward

        world.show()
        sleep(0.1)

    if world.reached_target:
        count_reached_targets += 1

print(count_reached_targets)