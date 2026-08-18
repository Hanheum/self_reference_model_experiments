import numpy as np
from agent import agent
from environment import gridworld

map_size = 5

world = gridworld(size=map_size)
Agent = agent(world_size=map_size)

epochs = 10000

train_start = 300

for epoch in range(epochs):
    terminated = False
    player_location, target_location = world.reset()
    Agent.reset_memory()

    total_reward = 0

    previous_player_location, previous_target_location = None, None
    previous_r = 0
    previous_action = None
    previous_terminated = False
    previous_memory_vector = None
    count = 0
    while not terminated:
        action = Agent.policy(np.concat([player_location, target_location]))
        new_player_location, new_target_location, reward, terminated = world.step(action)

        total_reward += reward

        if count != 0:
            Agent.memory.append([previous_player_location, previous_target_location, previous_memory_vector, previous_action, previous_r, Agent.q_value, previous_terminated])
        else:
            count += 1

        previous_player_location, previous_target_location = player_location.copy(), target_location.copy()
        previous_r = reward
        previous_action = action
        previous_terminated = terminated
        previous_memory_vector = Agent.memory_vector

        Agent.update_memory()
        player_location = new_player_location
        target_location = new_target_location

    Agent.memory.append([previous_player_location, previous_target_location, previous_memory_vector, previous_action, previous_r, 0, previous_terminated])

    if len(Agent.memory) >= train_start:
        loss = Agent.train()
        print(f"epoch: {epoch+1} | total reward: {total_reward} | memory length: {len(Agent.memory)} | epsilon: {Agent.epsilon} | loss: {loss} | reached target: {world.reached_target}")

    else:
        print(f"epoch: {epoch+1} | total reward: {total_reward} | memory length: {len(Agent.memory)} | epsilon: {Agent.epsilon} | reached target: {world.reached_target}")

    
    new_learning_rate = float(open('./learning_rate.txt', 'r').read())
    if Agent.model.learning_rate != new_learning_rate:
        Agent.model.learning_rate = new_learning_rate

    if (epoch+1)%100 == 0:
        Agent.model.save('./model_weight')