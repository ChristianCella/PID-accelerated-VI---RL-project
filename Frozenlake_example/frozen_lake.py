"""
Example of the frozen lake.

Reinforcement learning structure of the problem
-----------------------------------------------

While dynamic programming techniques can theoretically be applied to solve the Frozen Lake problem if the environment's dynamics are known, 
the typical usage of this environment in OpenAI Gym is for reinforcement learning experiments. 
The emphasis is on learning through interaction and feedback, which aligns with the principles of reinforcement learning rather than the 
model-based approach of dynamic programming.
In this case, we are in RL (control), and we are using the Q-learning algorithm to solve the problem: this is a off-policy method that chooses the greedy action:
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

In the code it is specified to choose whether you are training (you want to create the Q-table) or testing (you want to use the Q-table)
"""

import gymnasium as gym 
import numpy as np # To initialize the Q-table
import matplotlib.pyplot as plt # To plot the rewards
import pickle # To save the Q-table after the training is over

def run(episodes, is_training_true = True, render = False):
    
    # Set the environment and the q-table
    env = gym.make('FrozenLake-v1', map_name = '8x8', is_slippery = False, render_mode = 'human' if render else None) # Create the environment
    
    # If you are training: initilize the Q-table
    if(is_training_true):
        q = np.zeros((env.observation_space.n, env.action_space.n)) # Initialize the Q-table (64 states x 4 actions)
    else: # Load the q-table
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
                action = env.action_space.sample() # Random action (policy: it chooses a random action): 0 = left, 1 = down, 2 = right, 3 = up
            else: # Follow the q-table (when epsilon is zero or when I'm not training: the agent is acting greedily and only follows the q-table)
                action = np.argmax(q[state, :]) # for the specific state choose the action whose index corresponds to the maximum value in the q-table
                
            # Sample from the environment: this is the 'feedback' we get from the environment
            new_state, reward, terminated, truncated, _ = env.step(action)
            
            if(is_training_true):
                
                # After taking a step, update the q-table (only when training)
                q[state, action] = q[state, action] + learning_rate_a * (reward + discount_factor_g * np.max(q[new_state, :]) - q[state, action])
            
            # Update the state
            state = new_state
    
        epsilon = max(epsilon - epsilon_decy_rate, 0) # Decrease epsilon
        
        # Decide the learning rate
        if(epsilon == 0):
            learning_rate_a = 0.0001
            
        if reward == 1:
            rewards_per_episode[i] = 1
            
    env.close() # Close the environment
    
    sum_rewards = np.zeros(episodes)
    for t in range(episodes):
        sum_rewards[t] = np.sum(rewards_per_episode[max(0, t-100):(t + 1)]) # Show the rewards for every 100 episodes
        
    # Plot the rewards
    plt.plot(sum_rewards)
    plt.savefig('FrozenLake_example/frozen_lake8x8.png')
    plt.xlabel('Episodes')
    plt.ylabel('Number of rewards per 100 episodes')
    
    if(is_training_true):
        
        # Save the q_table only when training
        f = open("FrozenLake_example/frozen_lake8x8.pkl", "wb")
        pickle.dump(q, f)
        f.close()
        
    # print(f"The transition model of the environment is: {env.P}")
        
# test the environment
if __name__ == '__main__':
    
    # run(15000, is_training_true = True, render = False) # Training
    run(1, is_training_true = False, render = True) # After the training is complete
    
    