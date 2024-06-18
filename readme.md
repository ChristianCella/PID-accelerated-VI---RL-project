# RL project
## Structure
The project is subdivided in 4 main folders, 3 of which ([Frozenlake_example](https://github.com/ChristianCella/PID-accelerated-VI---RL-project/tree/main/Frozenlake_example), [src](https://github.com/ChristianCella/PID-accelerated-VI---RL-project/tree/main/src) and [tests](https://github.com/ChristianCella/PID-accelerated-VI---RL-project/tree/main/tests)) are not 'packages' (no '__init__.py'); for what concerns the first two folders, they are not directly related to the project. More specifically:
- [Frozenlake_example](https://github.com/ChristianCella/PID-accelerated-VI---RL-project/tree/main/Frozenlake_example), is a folder conatining a testbed for Reinforcement Learning using a tabular approach and some comparisons with Dynamic Programming and Deep Reinforcement Learning. For a better explanation of the codes, take a look at the [readme](https://github.com/ChristianCella/PID-accelerated-VI---RL-project/blob/main/Frozenlake_example/readme.md) inside the folder;
- [src](https://github.com/ChristianCella/PID-accelerated-VI---RL-project/tree/main/src) is the unmodified repo of the authors, added to the project just to have all the material as close as possible in case some changes need to be made.

## Tests and classes
The third and the fourth folder are the heart of the project.

The third folder, called [tests](https://github.com/ChristianCella/PID-accelerated-VI---RL-project/tree/main/tests), contains four files, each of which implements the code to obtain the corresponding figure in the paper.

The fourth folder, called [MDPs](https://github.com/ChristianCella/PID-accelerated-VI---RL-project/tree/main/MDPs), is actually a package and contains the classes ('FiniteMDP' and 'FrozenLakeMDP', defined in [finiteMDPs.py](https://github.com/ChristianCella/PID-accelerated-VI---RL-project/blob/main/MDPs/finiteMDPs.py)), the basic functions for the four different tests (defined in [vanilla_functions.py](https://github.com/ChristianCella/PID-accelerated-VI---RL-project/blob/main/MDPs/vanilla_functions.py)), the more complex functions relying on the basic ones (defined in [tests_functions.py](https://github.com/ChristianCella/PID-accelerated-VI---RL-project/blob/main/MDPs/tests_functions.py)) and also a file named [Rootlocus_functions.py](https://github.com/ChristianCella/PID-accelerated-VI---RL-project/blob/main/MDPs/Rootlocus_functions.py) that is used in some of the more complex functions to gain insight about the eigenvalues.
The last file of the folder is [Bellman_matrix_form.py](https://github.com/ChristianCella/PID-accelerated-VI---RL-project/blob/main/MDPs/Bellman_matrix_form.py) and it implements, for the case of 'Prediction' in Dynamic Programming, the matrix form of the Bellman Expectation Equation: in Prediction, the goal is to evaluate the Value function $V^\pi$, by knowing the policy in advance. For this reason, after evaluating the matrix $P^\pi$, the Bellman equation can be deterministically solved. This file is very useful because the form of the equation written by the original authors (function called 'value_iteration, inside [vanilla_functions.py](https://github.com/ChristianCella/PID-accelerated-VI---RL-project/blob/main/MDPs/vanilla_functions.py)') is not so easy to be interpreted, and you can have a double-check on the results.

## Some insight about the approaches presented in the paper

Reinforcement Learning (RL) and Dynamic Programming (DP) are not the same thing, but they can be seen as two different approaches to find the optimal solution to a problem. More specifically, the paper focuses on applying the Value Iteration algorithm in DP, but a clearer distinction must be made:
- if you want to evaluate the value function only and obtain $V^{\pi}$, then you are in the case of Prediction and you consider the policy $\pi$ to be deterministically known;
- if you also want to evaluate the optimal policy $\pi^{*}$ you can either use the Policy Iteration (PI) algorithm (the value of $\pi_k$ is calculatd at each iteration, together with $V_k$) or the Value Iteration algorithm (you iterate over $V_k$ and, once you reach $V^*$, you calculate $\pi^*$).
For the case of prediction, you can also explicitly calculate $V^{\pi}$ by solving the matrix form of the Bellman expectation equation $V^{\pi} = (I - \gamma P^{\pi})^{-1} \cdot R^{\pi}$. In this case, and only for this case, you can apply a procedure that is non-iterative, unlike the case of control that always requires an iteration over $V_k$.
In the paper, the solution of the matrix form was not present and an iterative procedure was presented in order to account both for Prediction and Control (that formula is not very easy to understand). However, by looking at [Bellman_matrix_form.py](https://github.com/ChristianCella/PID-accelerated-VI---RL-project/blob/main/MDPs/Bellman_matrix_form.py) it is possible to see how no iteration is required if you want to obtain $V^{\pi}$ directly.




