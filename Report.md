
## Report : DRLND Project 1



<br>


### Learning Algorithm
DQN is an algorithm created by DeepMind that brings together the power of the Q-Learning algorithm with the advantages of generalization through function approximation. It uses a deep neural network to estimate a Q-value function. As such, the input to the network is the current state of the environment, and the output is the Q-value for each action.

DeepMind also came up with two techniques that help successfully train a DQN agent:


- *Replay buffer*: Stores a finite collection of previous experiences that can be sampled randomly in the learning step. This helps break the correlation between consecutive actions, which leads to better conversion.
- *Separate target network*: Similar to the local network in structure, this network helps prevent large fluctuations in network updates, which leads to more stable training.

There have been multiple improvements to vanilla DQN suggested by RL researchers. In this project, I decided to implement a method called Double DQN. This method helps prevent the overestimation of Q-values, which is something that DQN agents are susceptible to. It does so by using the main network to pick the best action and the target network to evaluate that action for a given state. The idea here is that the two networks must agree that an action is good.

The DQN implementation for this project also includes soft updates of the target network. This is different from the periodic updates mentioned in the original DeepMind paper. This update mechanism uses the `TAU` hyperparameter. Other important hyperparameters include the `LEARNING_RATE`, `GAMMA` for discounted future rewards, `REPLAY_BUFFER_SIZE`, and the `BATCH_SIZE`, which controls the number of experiences sampled from the replay buffer in the learning step.

The model architecture used is the same for both the local and target networks. I used two hidden layers with 128 and 64 hidden units having ReLU activation and output layer with Linear Activation.

## Plot of Rewards
With `BUFFER_SIZE  =  2e5`, `BATCH_SIZE = 128`, `GAMMA = 0.90` agent solved the environment in 350 episodes with Vanilla DQN.
Performance of Double DQN was better as it showed low variance in average scores (over 100 episodes) and focused learning.DDQN solved the environment in 334 episodes.
<br>
<div align="center"><img src="https://github.com/iAbhyuday/Navigation-Deep-Q-Networks/raw/master/plot.png" ></div>

## Ideas for Future Work

To improve upon these results, my main idea is to implement a series of DQN improvements that have been suggested in recent years, including: prioritized experience replay, dueling DQNs, multi-step bootstrap targets, distributional DQN, and noisy DQN. A source of inspiration, and a good reading to get started towards this goal is DeepMind's Rainbow paper.
Another idea would be to try different agorithms and see how they perform against DQN. Simpler methods such as SARSA and Q-Learning could be used for this purpose, along with many other policy-based methods.
