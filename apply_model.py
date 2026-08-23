import numpy as np
from environment import gridworld
from agent import agent
from time import sleep

Agent = agent(world_size=5)
world = gridworld(5)

Agent.model.load('./model_weight')
Agent.epsilon = 0.

terminated = False
player_location, target_location = world.reset()
total_reward = 0

world.show()

player_location_input = input('start where:').split(', ')
player_location_input = list(map(int, player_location_input))
player_location = np.array(player_location_input)
world.player = player_location

while not terminated:
    action = Agent.policy(np.concat([player_location, target_location]), eval=True)
    player_location, target_location, reward, terminated = world.step(action)
    Agent.update_memory()
    total_reward += reward

    world.show()
    sleep(0.1)