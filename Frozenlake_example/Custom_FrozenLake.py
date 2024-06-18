""" 
In this code we will modify the FrozenLake environment to make it more suitable for the lewrning of the optimal policy.
PPo does not approximate the policy in a perfect way everywhere but, if correctly trained, it will approximnate the policy perfectly on the 
optimal path?
"""

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from gymnasium.envs.registration import register
from gymnasium.envs.toy_text.frozen_lake import FrozenLakeEnv

import numpy as np
import matplotlib.pyplot as plt
import pickle
import torch
from numpy.linalg import inv

import sys
sys.path.append('.')
import MDPs as mdp

class FrozenLake(FrozenLakeEnv):
    def __init__(self, desc = None,  map_name = '8x8', is_slippery = False, render_mode = None):
        super().__init__(desc = desc, map_name = map_name, is_slippery = is_slippery, render_mode = render_mode)
        
    def step(self, action):
        
        # Call the parent class method
        state, reward, done, truncated, info = super().step(action)
        
        # Modify the reward
        if state == 63:
            reward = 100
        elif state in self.hole_states:
            reward = -10
        else:
            reward = -1

        return state, reward, done, truncated, info
    
    # Indices that describe the holes
    @property
    def hole_states(self):
        return set([19, 29, 35, 41, 42, 46, 49, 52, 54, 59])
    

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
    learning_rate_a = 0.1 # Learning rate (alpha)
    
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
            #print(f"The reward is: {reward}")
            
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
    
# Function for Dynamic Programming    
def FrozenLakeDP(env, discount_factor_g):
 
    print(" ------------- Dynamic Programming: Value Iteration algorithm ------------- ")
    
    # Define the parameters for the Value Iteration algorithm
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    P_mat = env.P # Very complicated list of lists
    R = -np.ones((n_states, 1)) # Specific for the FrozenLake problem
    R[-1] = 100 # Only the last state has a reward of 1
    R[19] = -10
    R[29] = -10
    R[35] = -10
    R[41] = -10
    R[42] = -10
    R[46] = -10
    R[49] = -10
    R[52] = -10
    R[54] = -10
    R[59] = -10
    
    print(f"The vector of rewards in DP is {R}")
    
    # calculate the P tensor (one 'P' for each action)
    P = [np.matrix(np.zeros( (n_states, n_states))) for act in range(n_actions)]
            
    for state in range(n_states): 
        for action in range(n_actions):
            transitions = P_mat[state][action]
            for prob, next_state, reward, done in transitions:
                P[action][state, next_state] += prob
    
    print(f"The tensor P is: {P}")           
    # Call the function defined in 'vanilla_functions.py' (Control ==> Policy = None)
    _, q_optimal, _ = mdp.value_iteration(R, P, discount_factor_g, IterationsNo = None, policy = None)
    print(f"The optimal q-table obtained using DP is: {q_optimal}")
    
    # Now evaluate the optimal policy associated to that specific q-table
    policy_optimal = np.argmax(q_optimal, axis = 1)
    print(f"The optimal policy obtained in DP is: {policy_optimal.T}")
    
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
        ppo_mlp.learn(total_timesteps = 500000, log_interval = 8, progress_bar = True)
        ppo_mlp.save("FrozenLake_example/weightsNN")
    elif training == False:
        # Load the data (if you already have a trained model)
        model = ppo_mlp.load("FrozenLake_example/weightsNN") 
        env = FrozenLake(desc = None, map_name = '8x8', is_slippery = False, render_mode = 'human' if render else None)
        print(f"Loaded")
    
        # See what the agent is doing     
        print('...Using the agent...')

        # get the trained environment
        env = ppo_mlp.get_env() 

        for episode in range(1, episodes + 1):
            
            obs = env.reset()
            done = False
            score = 0

            while not done:
                
                env.render()
                action, _ = model.predict(obs) # The model predicts the action to take based on the observation. output: action and the value of the next state (used in recurrent policies)
                print(f"The selected action is: {action}")
                obs, reward, done, _ = env.step(action) # The environment takes the action and returns the new observation, the reward, if the episode is done, and additional information
                print(f"The reward is: {reward}")
                score += reward
                
            print(f'Episode: {episode}, Score: {score}')
        
    env.close()

    # Evaluate the optimal policy
    n_states = env.observation_space.n  # This assumes a discrete observation space
    optimal_policy = np.zeros(n_states)

    for s in range(n_states):
        action, _ = model.predict(s, deterministic = True)  # Get the action from the model
        optimal_policy[s] = action 
 
    #print("Policy vector (n_states x 1):")
    print(f"The optimal policy obtained with PPO is: {optimal_policy}")
        
    
if __name__ == '__main__':
    
    # Create an instance of the class
    render = True
    discount_factor_g = 0.6
    env = FrozenLake(desc = None, map_name = '8x8', is_slippery = False, render_mode = 'human' if render else None)
    
    # regsiter the environment
    register(
    id = "FrozenLake-v1",
    entry_point = "__main__:FrozenLake",
    max_episode_steps = 100, #  After 20 steps the episode terminates regardless if I reached or not  a terminal state
    reward_threshold = 0,
    )
    
    # Call the function
    # FrozenLakeRL(15000, training = True, env = env, discount_factor_g = discount_factor_g) 
    FrozenLakeRL(1, training = False, env = env, discount_factor_g = discount_factor_g)
    
    # Dynamic programming
    discount_factor_g = 0.6
    FrozenLakeDP(env, discount_factor_g)
    
    # Proximal Policy Optimization
    render = True
    env = FrozenLake(desc = None, map_name = '8x8', is_slippery = False, render_mode = 'human' if render else None)
    FrozenLakePPO(env = env, episodes = 1, learning_rate = 0.0003, training = False)
    
        