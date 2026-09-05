import numpy as np
from agent import agent
from environment import memory_using_task
from ml_utils import single_one_hot

map_size = 5

world = memory_using_task(size=map_size)
Agent = agent(world_size=map_size)

epochs = 10000

train_start = 30000
train_started = False

steps = 0

for epoch in range(epochs):
    terminated = False
    player_location, target_location, target_type = world.reset()
    target_type = single_one_hot(target_type, 4)
    Agent.reset_memory()

    total_reward = 0

    previous_memory = Agent.memory_vector.copy()
    epoch_step = 0
    total_loss = 0

    while not terminated:
        action = Agent.policy(np.concat([player_location, target_location, target_type]))
        new_player_location, new_target_location, reward, new_target_type, terminated = world.step(action)
        new_target_type = single_one_hot(new_target_type, 4)

        total_reward += reward

        Agent.memory.append([player_location, target_location, target_type, new_player_location, new_target_location, new_target_type, previous_memory, Agent.memory_vector, action, reward, terminated])

        Agent.update_memory()
        player_location = new_player_location
        target_location = new_target_location
        previous_memory = Agent.memory_vector

        if not train_started:
            if len(Agent.memory) >= train_start:
                train_started = True
        else:
            loss = Agent.train()
            total_loss += loss

        epoch_step += 1
        steps += 1

        if steps >= 300:
            steps = 0
            Agent.update_target_model()

    if train_started:
        print(f"epoch: {epoch+1} | total reward: {total_reward:.3f} | memory length: {len(Agent.memory)} | epsilon: {Agent.epsilon:.3f} | loss: {total_loss/epoch_step:.3f} | reached target: {world.reached_target} | right target: {world.is_right_spot}")

    else:
        print(f"epoch: {epoch+1} | total reward: {total_reward:.3f} | memory length: {len(Agent.memory)} | epsilon: {Agent.epsilon:.3f} | reached target: {world.reached_target} | right target: {world.is_right_spot}")

    new_learning_rate = float(open('./learning_rate.txt', 'r').read())
    if Agent.model.learning_rate != new_learning_rate:
        Agent.model.learning_rate = new_learning_rate

    if (epoch+1)%100 == 0:
        Agent.model.save('./model_weight')