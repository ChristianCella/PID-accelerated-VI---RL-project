# Frozen Lake example in detail

This folder can be used to understand Reinforcement Learning in a better way. Frozen lake is one of the environments avaialble in the OpenAI Gymnasium library and it is characterized by a discrete (finite) number of states and a discrete number of actions. Based on how it is defined, it is possible to implement simple Reinforcement Learning (RL) techniques (such as Q-learning), Dynamic Programming (DP) algorithms (i.e. value Iteration), and also Policy gradient-based methods such as Proximal Policy Optimization (PPO). The implementation revolves around the file called [Custom_FrozenLake.py](https://github.com/ChristianCella/PID-accelerated-VI---RL-project/blob/main/Frozenlake_example/Custom_FrozenLake.py), while the zip called [weights](https://github.com/ChristianCella/PID-accelerated-VI---RL-project/blob/main/Frozenlake_example/weightsNN.zip) contains the values of $\theta$ with which the policy $\pi(s, a, \theta)$ is parametrized.
 
## Main features

FrozenLake is a grid-based environment where the agent's objective is to navigate from a starting point (upper-left corner) to a goal (lower right) while avoiding holes and slippery surfaces. The state space consists of discrete positions on the grid, and the action space consists of discrete  movements (left = 0, down = 1, right = 2, up = 3). The default implementation of the algorithm is such that the reward is null everywhere, except in the final state that represents the reaching of the goal (notice that this 'sparse' reward makes the use of policy-gradient based methods such as PPO very difficult, since it is possible taht the agent does not learn the optimal policy).
The file [Custom_FrozenLake.py](https://github.com/ChristianCella/PID-accelerated-VI---RL-project/blob/main/Frozenlake_example/Custom_FrozenLake.py) copnatins the definition of a custom class for the implementation of FrozenLake: the custom nature is motivated by the need to modivfy the vector of reward in order to move from a 'sparse' setting to a more homogeneous configuration (for example, 100 when the goal is reached, -10 when the agent falls in the holes, -1 elsewhere).
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

In the second case, a DP approach is implemented instead: this entails the knowledge of the state-transition matrix (or better, tensor) P. It's important to decide between $\textbf{Prediction}$ (the goal is to evaluate 'how good' V is; this method is called Policy Evaluation) or $\textbf{Control}$ (the goal is to obtain $V^*$ and $\pi^*$). For the former the policy must be decided beforehand while, for the latter, the value of $\pi$ can be obtained iteratively (Policy Iteration = Policy Evaluation + Policy Improvement) or once the value of $V^*$ is known (Value Iteration).
For the case under exam, the idea is to verify that the Q table obtained in Reinforcement Learning (previous paragraph) is correct, by leveraging the function defined by the authors. In the end, by selecting a discount factor $\gamma$ = 0.6, it is possible to verify that the Q tables are similar. It's important to notice though that different Q-tables can lead to the same optimal policy, that is obtained bt selecting, row by row, the action (column) that corresponds to the highest probability (meaning of 'acting greedily', tyipical of Control): it can happen that as a function of $\gamma$ the results can be a little different (In terms of Q-tables).

## Third approach: Proximal Policy Optimization

Proximal Policy Optimization is a Policy Gradient-based algorithm with which it is possible to approximate both the value function and the policy (actor-criitic method), provided that the function approximators are suitable (in this case, the area is Deep Reinforcement Learning). In this case, both the value function and the policy are approximated with neural networks characterized by 2 hidden layers (responsible for transforming the input into a more abstract representation) featured by 32 units (neurons) each. Each layer applies a linear transformation followed by a non-linear activation function (e.g. ReLU).
The idea behind PPO s to maximize the expected advantage while avoiding large policy updates. This is achieved using a clipped surrogate objective:

$L^{CLIP}(\theta) = \mathbb{E}_t[\text{min}(r_t(\theta)\hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t)]$

where:

- $r_t(\theta) = \frac{\pi_{\theta}(a_t|s_t)}{\pi_{\theta, old}(a_t|s_t)}$ is the probability ratio;
- $\hat{A}_t$ is the advantage estimation at time t, taht can be computed by the Generalized Advantage Estimation (GAE) $\rightarrow$ $\hat{A}_t = \delta_t + (\gamma \lambda)\delta_{t+1} + (\gamma \lambda)^2 \delta_{t+1} + \cdots$. More in detail:
    - $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t) \rightarrow$ TD residual;
    - $\gamma \rightarrow$ discount factor;
    - $\lambda \rightarrow$ GAE parameter.

The optimal policy $\pi$ is obtained by updating iteratively the weights $\theta$ of the neural network used to approximate $\pi \rightarrow \pi_{\theta}(s, a, \theta)$_

$\theta_{k+1} = \text{argmax}_{\theta}\mathbb{E}_t[L^{CLIP}(\theta) - c_1\mathbb{E}_t[(V_{\theta}(s_t) - V_t^{\text{target}})^2] + c_2\mathbb{E}_t[\text{Entropy}[\pi_{\theta}](s_t)]]$

where:

- $c_1$ and $c_2$ are coefficients for value function loss and entropy bonus, respectively;
- $V_t^{\text{target}}$ is the target value $\rightarrow \hat{R}_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \cdots$
- $\text{Entropy}[\pi_{\theta}](s_t) = -\sum_{a}\pi_{\theta}(a|s_t)\text{log}\pi_{\theta}(a|s_t) \rightarrow$ entropy term added to encourage exploration and to penalize deterministic policies.

One important remark must be made: in FrozenLake, there is more than one optimal path composed of 15 steps that allow the agent to reach the goal. In the code, Q-learning allows to reach the end passing through the states 0, 8, 16, 17, 18, 26, 27, 28, 36, 37, 38, 39, 47, 55, 63, while PPO (after a training of 500000 time steps) finds a path connecting the states 0, 1, 9, 10, 11, 12, 13, 21, 22, 30, 38, 39, 47, 55, 63. The weights $\theta$ are stored inside [weights](https://github.com/ChristianCella/PID-accelerated-VI---RL-project/blob/main/Frozenlake_example/weightsNN.zip).





