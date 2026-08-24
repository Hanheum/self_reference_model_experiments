Experiments over an architecture which refers to its previous states on its current state.

Currently this repository contains only one experiment, whch is executed by experiment.py, which is about 5x5 Gridworld DQN. 
The model used in the experiment was different from simple Dense layers because it's input layer is concat of observation and specific layer that's involved in the propagation. 
Which can be interpreted as model refering to its previous action. I made premise that this can lead to a behavior similar to memory. 

When experiment.py is executed, it will run gridworld environment and save its result in Agent.memory. After it collects enough samples, it will autumatically perform training.
Every 100 epochs(episodes), model's weights and bias arrays will be saved. 
And when apply_model.py is executed, saved weights and bias will be loaded, and run a single episode with no random actions. (Completely greedy algorithm)

The model is proved to be good enough for performing simple gridworld task. 
Later, it will be tested against harder projects that actually require memory in order to earn rewards.
(Such as an environment where enemies follow the player and it must collect some objects in order to survive.)

Contact chh3653@gmail.com for more information.