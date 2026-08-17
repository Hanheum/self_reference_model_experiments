import numpy as np
from agent import agent
from environment import gridworld

map_size = 5

world = gridworld(size=map_size)
Agent = agent(world_size=map_size)

epochs = 10000

train_start = 30000

for epoch in range(epochs):
    terminated = False
    player_location, target_location = world.reset()
    Agent.reset_memory()

    total_reward = 0

    while not terminated:
        action = Agent.policy(np.concat([player_location, target_location]))
        new_player_location, new_target_location, reward, terminated = world.step(action)

        total_reward += reward

        Agent.memory.append([player_location, target_location, Agent.memory_vector, action, reward, terminated])

        Agent.update_memory()
        player_location = new_player_location
        target_location = new_target_location

    if len(Agent.memory) >= train_start:
        loss = Agent.train()
        print(f"epoch: {epoch+1} | total reward: {total_reward} | memory length: {len(Agent.memory)} | epsilon: {Agent.epsilon} | loss: {loss} | reached target: {world.reached_target}")

    else:
        print(f"epoch: {epoch+1} | total reward: {total_reward} | memory length: {len(Agent.memory)} | epsilon: {Agent.epsilon} | reached target: {world.reached_target}")

    
    new_learning_rate = float(open('./learning_rate.txt', 'r').read())
    if Agent.model.learning_rate != new_learning_rate:
        Agent.model.learning_rate = new_learning_rate