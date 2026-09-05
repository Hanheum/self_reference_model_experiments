import numpy as np
from environment import memory_using_task
from agent import agent
from time import sleep
from ml_utils import single_one_hot

Agent = agent(world_size=5)
world = memory_using_task(5)

Agent.model.load('./model_weight')
Agent.epsilon = 0.

terminated = False
player_location, target_location, target_type = world.reset()
total_reward = 0

raw_input = input('player start where:')
if raw_input != '':
    player_location_input = raw_input.split(', ')
    player_location_input = list(map(int, player_location_input))
    player_location = np.array(player_location_input)
    world.player = player_location

raw_input = input('target where:')
if raw_input != '':
    target_location_input = raw_input.split(', ')
    target_location_input = list(map(int, target_location_input))
    target_location = np.array(target_location_input)
    world.target = target_location

world.init_show()
sleep(0.2)

while not terminated:
    action = Agent.policy(np.concat([player_location, target_location, single_one_hot(target_type, 4)]), eval=True)
    player_location, target_location, reward, target_type, terminated = world.step(action)
    Agent.update_memory()
    total_reward += reward

    world.show()
    sleep(0.2)

print(f"total_reward: {total_reward}")