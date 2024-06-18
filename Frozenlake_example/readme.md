# Frozen Lake example in detail

This folder can be used to understand Reinforcement Learning in a better way. Frozen lake is one of the environments avaialble in the OpenAI Gymnasium library and it is characterized by a discrete (finite) number of states and a discrete number of actions. Based on how it is defined, it is possible to implement simple Reinforcement Learning (RL) techniques (such as Q-learning), Dynamic Programming (DP) algorithms (i.e. value Iteration), and also Policy gradient-based methods such as Proximal Policy Optimization (PPO). The implementation revolves around the file called [Custom_FrozenLake.py](https://github.com/ChristianCella/PID-accelerated-VI---RL-project/blob/main/Frozenlake_example/Custom_FrozenLake.py), while the zip called [weights](https://github.com/ChristianCella/PID-accelerated-VI---RL-project/blob/main/Frozenlake_example/weightsNN.zip) contains the values of $\theta$ with which the policy $\pi(s, a, \theta)$ is parametrized.
 
## Main features

FrozenLake is a grid-based environment where the agent's objective is to navigate from a starting point (upper-left corner) to a goal (lower right) while avoiding holes and slippery surfaces. The state space consists of discrete positions on the grid, and the action space consists of discrete  movements (left = 0, down = 1, right = 2, up = 3). The default implementation of the algorithm is such that the reward is null everywhere, except in the final state that represents the reaching of the goal (notice that this 'sparse' reward makes the use of policy-gradient based methods such as PPO very difficult, since it is possible taht the agent does not learn the optimal policy).
The file [Custom_FrozenLake.py](https://github.com/ChristianCella/PID-accelerated-VI---RL-project/blob/main/Frozenlake_example/Custom_FrozenLake.py) copnatins the definition of a custom class for the implementation of FrozenLake: the custom nature is motivated by the need to modify the vector of reward in order to move from a 'sparse' setting to a more homogeneous configuration (for example, 100 when the goal is reached, -10 when the agent falls in the holes, -1 elsewhere).
In addition, three different functions are defined, each of which implements one of the three qlgorithms mentioned above (RL, DP, PPO).

## First approach: Reinforcement Learning

In FrozenLake, the most straightforward approach is to let the agent interact with the environment to learn the optimal policy $\pi^*$ to reach the goal. In this case (RL), a new action must be sampled, since the hypothesis at the base is that the state-transition model $P^{\pi}$ is not known. The emphasis is on learning through 'interaction-and-feedback', which aligns with the principles of reinforcement learning rather than the model-based approach of dynamic programming. The first function, 'FrozenLakeRL', allows to implement the Q-learning algorithm (RL, control problem), that is the counterpart of Value Iteration for Dynamic Progarmming. With this function you can decide whether you are in $training$ (you want to learn the q-table) or if you want to use the learned q-table to reach the goal:
- in training, you want to determine the q-table using the following formula:

    $Q(s_t, a_t) = Q(s_t, a_t) + \alpha \cdot [r_t + \gamma \cdot max_aQ(s_{t+1}, a_{t+1}) - Q(s_t, a_t)]$

    in this sense, you need to choose $\alpha$ so as to rspect the Robins-Monroe conditions:

    $\sum_{t=1}^{\infty} \alpha_t = \infty$, $\sum_{t=1}^{\infty} \alpha_t^2 < \infty$

    For the project, the following hyperparameters were selected: $\epsilon_0$ = 1, $\alpha$ = 0.1, $\gamma$ = 0.6. After the training, the image [frozen_lake8x8.png](https://github.com/ChristianCella/PID-accelerated-VI---RL-project/blob/main/Frozenlake_example/frozen_lake8x8.png) is produced and shows that after some time the agent learns one of the optimal path to reach the goal (15 steps, passing by the states 0, 8, 16, 17, 18, 26, 27, 28, 36, 37, 38, 39, 47, 55, 63).
- If you want to visualize the result, youcan simply run one episode.
- As an example, assume this case: the Q-table at line 62, for example, is as follows:
//   0    1    2    3 ==> actions
.|  _     _    _    _
.|  _     _    _    _
.|  _     _    _    _
62| 0.1  0.12  0.3  0.2
63|  _     _    _    _

    'Acting greedyly' means that the agent chooses the action with the highest Q-value. In this case, the agent is in state 62 and the 
    action with the highest Q-value is 2 = 'go right' (0.3).




## Second approach: Dynamic Programming

In the second case, a DP approach is implemented instead, and we can since we know P, we should decide between Prediction (I want to know 
how good V is) or control (I want to obtain V* and pi*). In case of Prediction we need to decide a policy pi beforehand, while in case of Control we can
obtain the value of pi either iteratively (policy iteration = policy evaluation + policy improvement) or once the value of V* is known (value iteration).
 

* The example is a 8x8 grid, for a total of 64 states (from 0 = upper left corner to 63 = lower right corner).
* The guy receives a reward of 1 when it reaches the final state (63) and 0 otherwise.
* 'is_slippery', when set to True, is such that the method 'step' does not honor the chosen action with probability 1.0, but with 
    probability 0.33 it will move in a direction perpendicular to the chosen direction ==> this is a randomization.
* Goal of Q-learning: I want to build a q-table to solve the problem ==> this table is built DURING training.




