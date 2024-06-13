# Frozen Lake example in detail

This folder can be used to understand Reinforcement Learning in a better way. Frozen lake is one of the environments avaialble in OpenAI Gymnasium library and it is characterized by a discrete (finite) number of states and a discrete number of actions. Based on how it is defined, you can implement both Reinforcement Learning (RL)techniques and Dynamic Programming (DP) techniques.

## main characteristics

FrozenLake is a grid-based environment where the agent's objective is to navigate from a starting point (upper-left corner) to a goal (lower right) while avoiding holes and slippery surfaces. The state space consists of discrete positions on the grid, and the action space consists of discrete  movements (left = 0, down = 1, right = 2, up = 3). The default implementation of teh algorithm is such that the reward is null everywhere, except in the final state that represents the reaching of the goal (notice that this 'sparse' reward makes the use of policy-gradient based methods such as PPO very difficult).

## First approach: Reinforcement Learning
In this environment, the most straightforward approach is to let the agent interact with the environment to learn the optimal policy to reach the goal. In this case (RL) you must 'sample' a new action, because you are supposing that you do not know the state-transition model $P^{\pi}$. The emphasis is on learning through 'interaction-and-feedback', which aligns with the principles of reinforcement learning rather than the model-based approach of dynamic programming. The first function,'FrozenLakeRL', allows to implement the Q-learning algorithm (RL, control problem), that is the counterpart of Value Iteration for Dynamic Progarmming. With this function you can decide whether you are in training (you want to learn the q-table) or if you want to use the learned q-table to reach the goal:
- in training, you want to determine the q-table using the following formula Q(s, a) = Q(s, a) + $\gamma$ * (r + $\gamma$ * $max_a$(Q(s', a) - Q(s, a)))


## Second approach: Dynamic Programming

NOTE: if we wanted to adpot dynamic programming to solve this problem, and we can since we know P, we should decide between Prediction (I want to know 
how good V is) or control (I want to obtain V* and pi*). In case of Prediction we need to decide a policy pi beforehand, while in case of Control we can
obtain the value of pi either iteratively (policy iteration = policy evaluation + policy improvement) or once the value of V* is known (value iteration).
 

In this case, we are in RL (control), and we are using the Q-learning algorithm to solve the problem: this is a 'off-policy' method that chooses the greedy action:
    * This is the same as Bellman Optimality operator applied in case of Value Iteration (DP).
    * The update rule is the following: Q(s, a) = (1 - alpha) * Q(s, a) + alpha * (r + gamma * max(Q(s', a')) - Q(s, a))

* The example is a 8x8 grid, for a total of 64 states (from 0 = upper left corner to 63 = lower right corner).
* The guy receives a reward of 1 when it reaches the final state (63) and 0 otherwise.
* 'is_slippery', when set to True, is such that the method 'step' does not honor the chosen action with probability 1.0, but with 
    probability 0.33 it will move in a direction perpendicular to the chosen direction ==> this is a randomization.
* Goal of Q-learning: I want to build a q-table to solve the problem ==> this table is built DURING training.

IMPORTANT
---------
Assume this case: the Q-table at line 62, for example, is as follows:
//   0    1    2    3 ==> actions
.|  _     _    _    _
.|  _     _    _    _
.|  _     _    _    _
62| 0.1  0.12  0.3  0.2
63|  _     _    _    _

'Acting greedyly' means that the agent chooses the action with the highest Q-value. In this case, the agent is in state 62 and the 
action with the highest Q-value is 2 = 'go right' (0.3).

In the code it is specified to choose whether you are training (you want to create the Q-table) or testing (you want to use the Q-table); for the 
training part you also have the possibility to compare the results obtained in Reinforcement Learning with thos you would obtain implementing 
the Value Iteration algorithm for the case of control (policy = None).

Some comparisons are also done with a specific policy-gradient method, called PPO (Proximal Policy Optimization), that
is also what ChatGPT uses.
