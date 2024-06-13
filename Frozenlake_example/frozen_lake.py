"""
Example of the frozen lake.

The FrozenLake environment is a grid-based environment where the agent's objective is to navigate from a starting point to a goal while avoiding 
holes and slippery surfaces. The state space consists of discrete positions on the grid, and the action space consists of discrete 
movements (left, right, up, down).

While dynamic programming techniques can theoretically be applied to solve the Frozen Lake problem if the environment's dynamics are known (and they actually are), 
the typical usage of this environment in OpenAI Gym is for reinforcement learning experiments (like in this code).

NOTE: if we wanted to adpot dynamic programming to solve this problem, and we can since we know P, we should decide between Prediction (I wnat to know 
how good V is) or control (I want to obtain V* and pi*). In case of Prediction we need to decide a policy pi beforehand, while in case of Control we can
obtain the value of pi either iteratively (policy iteration = policy evaluation + policy improvement) or once the value of V* is known (value iteration).
 
The emphasis is on learning through 'interaction-and-feedback', which aligns with the principles of reinforcement learning rather than the 
model-based approach of dynamic programming.

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
"""

import gymnasium as gym 
import numpy as np # To initialize the Q-table
import matplotlib.pyplot as plt # To plot the rewards
import pickle # To save the Q-table after the training is over
 
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
import torch

import sys
sys.path.append('.')
# Import the package conatining all funcrions and classes to be used
import MDPs as mdp

def run(episodes, is_training_true = True, render = False, comparison = False):
    
    # Set the environment and the q-table
    env = gym.make('FrozenLake-v1', map_name = '8x8', is_slippery = False, render_mode = 'human' if render else None) # Create the environment
    
    # If you are training: initilize the q-table (in training you want to fill this)
    if(is_training_true):
        q = np.zeros((env.observation_space.n, env.action_space.n)) # Initialize the Q-table (64 states x 4 actions)
    else: # Load the q-table, created during training
        f = open('FrozenLake_example/frozen_lake8x8.pkl', 'rb')
        q = pickle.load(f)
        f.close()
           
    # Set the hyperparameters
    learning_rate_a = 0.9 # Learning rate (alpha)
    discount_factor_g = 0.9 # Discount factor (gamma)
    
    # Implement the epsilon-greedy policy
    epsilon = 1 # I'm picking 100% random actions
    epsilon_decy_rate = 0.0001 # epsilon decreases by 0.0001 at each episode (I reduce the randomness with time) ==> This impacts the minimum number of episodes
    # required for the training to be effective: if I want epsilon to go to 0, I need to train for more than 10,000 episodes.
    rng = np.random.default_rng() # Random number generator (here, you are just creating an instance of the random number generator)
    
    # keep track of the reward
    rewards_per_episode = np.zeros(episodes)
    
    # Loop over all episodes
    for i in range(episodes):
        
        state = env.reset()[0] # Reset the environment and get the initial state ([0])
        terminated = False # Flag to check if the episode is terminated (or the final state is reached)
        truncated = False # True when actions > 200 (the human is wandering around without reaching the final state)
        
        while(not terminated and not truncated): # Loop until the episode is terminated or the number of actions is greater than 200
            
            rand_numb = rng.random() # Generate a random number between 0 and 1
            
            if is_training_true and rand_numb < epsilon: # Random action  
                
                # Random action (policy: it chooses a random action): 0 = left, 1 = down, 2 = right, 3 = up
                # This 'sampling' is synonimous with RL: the agent is exploring the environment (assuming we do not know the environment dynamics, even though we do)   
                action = env.action_space.sample() 
                
            else: 
                
                # Follow the q-table (when epsilon is zero or when I'm not training: the agent is acting 'greedily' and only follows the q-table)
                action = np.argmax(q[state, :]) # for the specific state choose the action whose index corresponds to the maximum value in the q-table
                
            # Sample from the environment: this is the 'feedback' we get from the environment
            new_state, reward, terminated, truncated, _ = env.step(action)
            
            if(is_training_true):
                
                # After taking a step, update the q-table (only when training)
                q[state, action] = q[state, action] + learning_rate_a * (reward + discount_factor_g * np.max(q[new_state, :]) - q[state, action])
                # Q[:, act] = R + discount * P[act] * V # 50x4 matrix
            # Update the state
            state = new_state
    
        epsilon = max(epsilon - epsilon_decy_rate, 0) # Decrease epsilon = decrease the exploration
        
        # Decide the learning rate
        if(epsilon == 0):
            learning_rate_a = 0.0001
            
        if reward == 1:
            rewards_per_episode[i] = 1
    
    sum_rewards = np.zeros(episodes)
    for t in range(episodes):
        sum_rewards[t] = np.sum(rewards_per_episode[max(0, t - 100) : (t + 1)]) # Show the rewards for every 100 episodes
        
    # Plot the rewards
    plt.plot(sum_rewards)
    plt.savefig('FrozenLake_example/frozen_lake8x8.png')
    plt.xlabel('Episodes')
    plt.ylabel('Number of rewards per 100 episodes')
    
    if is_training_true:
        
        # Save the q_table only when training
        f = open("FrozenLake_example/frozen_lake8x8.pkl", "wb")
        pickle.dump(q, f)
        f.close()
        
    # print(f"The transition model of the environment is: {env.P}")
    print(f"The q-table learned in Reinforcement Learning is: {q}")
    
    # verify that using Value Iteration, the q-table is the same
    if comparison:

        # Define the parameters for the Value Iteration
        n_states = env.observation_space.n
        n_actions = env.action_space.n
        P_mat = env.P # Very complicated list of lists
        R = np.zeros((n_states, 1)) # Specific for the FrozenLake problem
        R[-1] = 1 # Only the last state has a reward of 1
       
        # calculate the P tensor
        P = [np.matrix(np.zeros( (n_states, n_states))) for act in range(n_actions)]
               
        for state in range(n_states):
            for action in range(n_actions):
                transitions = P_mat[state][action]
                for prob, next_state, reward, done in transitions:
                    P[action][state, next_state] += prob
                   
        # Call the function defined in 'vanilla_functions.py' (Control ==> Policy = None)
        _, q_optimal, _ = mdp.value_iteration(R, P, discount_factor_g, IterationsNo = None, policy = None)
        print(f"The optimal q-table obtained using DP is: {q_optimal}")
        
        # Now evaluate the optimal policy associated to that specific q-table
        policy_optimal = np.argmax(q_optimal, axis = 1)
        print(f"The optimal policy is: {policy_optimal}")

        if torch.cuda.is_available():
            device = torch.device('cuda')
            print("CUDA is available. Using GPU.")
        else:
            device = torch.device('cpu')
            print("CUDA is not available. Using CPU.")
        device = torch.device("cuda")        
        # To have a more refined comparison, try to see if PPO (actor-critic method) gives the same policy      
        ppo_mlp = PPO("MlpPolicy", env, verbose=1,
                   learning_rate = 0.001,
                   device = device,
                   policy_kwargs=dict(net_arch = [dict(pi = [32, 32], vf = [32, 32])]))
        
        # Train the model and save the data
        
        #ppo_mlp.learn(total_timesteps = 1000000, log_interval = 8, progress_bar = True)
        #ppo_mlp.save("FrozenLake_example/ppo_mlp_optimal_policy")
        
        ppo_mlp.load("FrozenLake_example/ppo_mlp_optimal_policy")
        env = ppo_mlp.get_env()
        # Extract the policy          
        #policy_vector = np.zeros(n_states, dtype=int)  
        
        n_states = env.observation_space.n  # This assumes a discrete observation space
        optimal_policy = np.zeros(n_states)

        """ 
        for s in range(n_states):
            # env.reset()
            #env.env.state = s  # Set the environment to the specific state
            action, _ = ppo_mlp.predict(s, deterministic = True)  # Get the action from the model
            optimal_policy[s] = action 
        """
        

        print('Using the agent')
        episodes = 5
        for episode in range(1, episodes + 1):
            obs = env.reset()
            print(f"the type of obs is {type(obs)}")
            print(f"obs: {obs}")
            done = False
            score = 0

            while not done:
                env.render()
                action, _ = ppo_mlp.predict(obs) # The model predicts the action to take based on the observation. output: action and the value of the next state (used in recurrent policies)
                print(f"action: {action}")
                obs, reward, done, info = env.step(action) # The environment takes the action and returns the new observation, the reward, if the episode is done, and additional information
                print(f"the reward is {reward}")
                score += reward
                
            print(f'Episode: {episode}, Score: {score}')

            
        #print("Policy vector (n_states x 1):")
        #print(optimal_policy)
        
    env.close() # Close the environment
                

    
        
# test the environment
if __name__ == '__main__':
    
    # run(15000, is_training_true = True, render = False, comparison = True) # Training
    run(1, is_training_true = False, render = True, comparison = True) # After the training is complete
    
    