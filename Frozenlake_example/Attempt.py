""" 
Final version of the Frozenlake example.
"""

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

import numpy as np
import matplotlib.pyplot as plt
import pickle
import torch

import sys
sys.path.append('.')
import MDPs as mdp

# Function for Reinforcement Learning
def FrozenLakeRL(episodes, training, env, discount_factor_g):
   
    print(" ------------- Reinforcement Learning: Q-learning ------------- ")
   
    # If you are training: initilize the q-table (in training you want to fill this)
    if(training):
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
    epsilon_decay_rate = 0.0001 # epsilon decreases by 0.0001 at each episode (I reduce the randomness with time) ==> This impacts the minimum number of episodes
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
            
            if training and rand_numb < epsilon: # Random action                 
                # Random action (policy: it chooses a random action): 0 = left, 1 = down, 2 = right, 3 = up
                # This 'sampling' is synonimous with RL: the agent is exploring the environment (assuming we do not know the environment dynamics, even though we do)   
                action = env.action_space.sample()                
            else:                 
                # Follow the q-table (when epsilon is zero or when I'm not training: the agent is acting 'greedily' and only follows the q-table)
                action = np.argmax(q[state, :]) # for the specific state choose the action whose index corresponds to the maximum value in the q-table
                
            # Sample from the environment: this is the 'feedback' we get from the environment
            new_state, reward, terminated, truncated, _ = env.step(action)
            
            if(training):              
                # After taking a step, update the q-table (only when training)
                q[state, action] = q[state, action] + learning_rate_a * (reward + discount_factor_g * np.max(q[new_state, :]) - q[state, action])

            # Update the state
            state = new_state
    
        epsilon = max(epsilon - epsilon_decay_rate, 0) # Decrease epsilon = decrease the exploration
        
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
    
    if training:
        
        # Save the q_table only when training
        f = open("FrozenLake_example/frozen_lake8x8.pkl", "wb")
        pickle.dump(q, f)
        f.close()
    
    # Print the q-table obtained in Reinforcement Learning
    print(f"The q-table learned in Reinforcement Learning is: {q}")
        
    env.close()

# Function for Dynamic Programming    
def FrozenLakeDP(env, discount_factor_g):
    
    print(" ------------- Dynamic Programming: Value Iteration algorithm ------------- ")
    
    # Define the parameters for the Value Iteration algorithm
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    P_mat = env.P # Very complicated list of lists
    R = np.zeros((n_states, 1)) # Specific for the FrozenLake problem
    R[-1] = 1 # Only the last state has a reward of 1
    
    # calculate the P tensor (one 'P' for each action)
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
    print(f"The optimal policy obtained in DP is: {policy_optimal.T}")

# Function that implements the Proximal Policy Optimization (PPO) algorithm
def FrozenLakePPO(env, episodes, learning_rate, training):
  
    #
    # env = gym.make('FrozenLake-v1', map_name = '8x8', is_slippery = False, render_mode = 'human')
    
    # Possibly move the training on the GPU
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print("CUDA is available. Using GPU.")
    else:
        device = torch.device('cpu')
        print("CUDA is not available. Using CPU.")
        
    # Creation of an instance of the class PPO     
    ppo_mlp = PPO("MlpPolicy", env, verbose = 1,
                learning_rate = learning_rate,
                device = device,
                policy_kwargs=dict(net_arch = [dict(pi = [32, 32], vf = [32, 32])]))
       
    
    if training:
         
        # Train the model and save the data  
        ppo_mlp.learn(total_timesteps = 1000000, log_interval = 8, progress_bar = True)
        ppo_mlp.save("FrozenLake_example/ppo_mlp_optimal_policy")
    else:
        # Load the data (if you already have a trained model)
        ppo_mlp.load("FrozenLake_example/ppo_mlp_optimal_policy") 
    
    # See what the agent is doing     
    print('...Using the agent...')
    env = ppo_mlp.get_env() 
    
    for episode in range(1, episodes + 1):
        
        obs = env.reset()
        done = False
        score = 0

        while not done:
            
            env.render()
            action, _ = ppo_mlp.predict(obs) # The model predicts the action to take based on the observation. output: action and the value of the next state (used in recurrent policies)
            print(f"The selected action is: {action}")
            obs, reward, done, info = env.step(action) # The environment takes the action and returns the new observation, the reward, if the episode is done, and additional information
            print(f"The reward is: {reward}")
            score += reward
            
        print(f'Episode: {episode}, Score: {score}')

    # Evaluate the optimal policy
    n_states = env.observation_space.n  # This assumes a discrete observation space
    optimal_policy = np.zeros(n_states)

    for s in range(n_states):
        action, _ = ppo_mlp.predict(s, deterministic = True)  # Get the action from the model
        optimal_policy[s] = action 
 
    #print("Policy vector (n_states x 1):")
    print(f"The optimal policy obtained with PPO is: {optimal_policy}")

if __name__ == '__main__':
    
    # Create the environment
    render = True
    discount_factor_g = 0.9
    env = gym.make('FrozenLake-v1', map_name = '8x8', is_slippery = False, render_mode = 'human' if render else None)
    
    
    # Reinforcement learning
    # FrozenLakeRL(15000, training = True, env = env, discount_factor_g) # Training (set 'render' to False)
    FrozenLakeRL(1, training = False, env = env, discount_factor_g = discount_factor_g) # After the training (set 'render' to True)

    # Dynamic programming
    FrozenLakeDP(env, discount_factor_g)
    
    # Proximal Policy Optimization
    episodes = 5
    learning_rate = 0.001
    training = False
    render = True
    discount_factor_g = 0.9
    env = gym.make('FrozenLake-v1', map_name = '8x8', is_slippery = False, render_mode = 'human' if render else None)
    FrozenLakePPO(env = env, episodes = episodes, learning_rate = learning_rate, training = training)
    