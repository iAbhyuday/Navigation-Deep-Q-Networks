
## Deep Reinforcement Learning ND Project 01 : Navigation
<div align="center"><img src="https://github.com/iAbhyuday/Navigation-Deep-Q-Networks/raw/master/banana.gif" ></div>


<br><br>


### Project Details
This project uses Deep Q Network to train an RL-Agent to navigate (and collect bananas!) in a large, square world.
<u style="text-decoration-color:gray">A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.</u> Thus, the goal of the RL agent is to collect as many yellow bananas as possible while avoiding blue bananas.

<u style="text-decoration-color:gray">The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around the agent's forward direction</u>. Given this information, the agent has to learn how to best select actions. Four discrete actions are available, corresponding to:

   - **`0`**  move forward
   - **`1`**  move backward
   - **`2`**  turn left
   - **`3`**  turn right
<br>
The task is episodic, and in order to solve the environment, agent must get an average score of +13 over 100 consecutive episodes.
### Getting Started
- Clone this repository
- Download the Unity Environment
   - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
- Unzip in Repo folder
- Create a virtual environment with python3.6
- Install dependencies via pip 
  ```
  pip intall -r requirements.txt
  ```
### Instructions
1. Change the location for Unity Environment in  **`Navigation.ipynb`** at cell 4
2. Test the pretrained model by function in cell 13
3. Feel free to change the hyperparameters of the agent and the Network Architecture 
4. Explore.
   
