from environment import memory_using_task
from random import randint
from time import sleep

world = memory_using_task()
terminated = False

player_location, target_location = world.reset()
print(player_location, target_location)
print(world.spots)

total_reward = 0

world.init_show()

while not terminated:
    action = randint(0, 3)
    new_player_location, new_target_location, reward, target_type, terminated = world.step(action)

    total_reward += reward

    print(f"target type: {target_type}, reached_target: {world.reached_target}, is_right_spot: {world.is_right_spot}")
    world.show()

    sleep(0.2)