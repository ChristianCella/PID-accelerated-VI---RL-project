""" 
This script allows to obatin the figures 3 of the paper, depending on the policy that is specified.
"""

# Import standard libraries
from collections import namedtuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import *

import sys 
sys.path.append('.')
# Import the package conatining all funcrions and classes to be used
import MDPs as mdp

# Define the parameters of the problem
discount = 0.99
iter_no = 2000
Gain = namedtuple('Gain', ['kP', 'kI', 'kD', 'I_alpha', 'I_beta'], defaults = [1, 0, 0, 0, 0])
error_norm_ord = np.inf # or 2 or 1 or np.inf
discount_str = str(discount).replace('.','p')
meta_lr = 0.05
normalization_eps = 1e-20
hyperparam_detail = '(eta,eps)=(' + str(meta_lr).replace('.','p') +\
                    ',' + str(normalization_eps).replace('.','p')+')'
gain_list = [Gain(1.0, 0, 0, 0.05, 0.95)]

# Define the problem to be analyzed
ProblemType = 'FrozenLake'

# Create an instance of the correct class
if ProblemType == 'randomwalk':
    
    # Problem-specific parameters
    state_size = 50
    
    # Select the policy
    # pi = [0] * state_size # Policy that always chooses the first action
    # pi = np.random.randint(0, 2, state_size) # Random policy
    pi = None # For the Bellman optimality
    
    # Chain walk problem
    MDP = mdp.FiniteMDP(state_size, ProblemType = 'randomwalk')
    
elif ProblemType == 'garnet':
    
    # Problem-specific parameters
    state_size = 50
    action_size = 4
    GarnetParam = (3,5)
    
    # Select the policy
    # pi = [0] * state_size # Policy that always chooses the first action
    # pi = np.random.randint(0, 2, state_size) # Random policy
    pi = None # For the Bellman optimality
    
    # Garnet random problem
    MDP = mdp.FiniteMDP(state_size, ActionSize = action_size, ProblemType = 'garnet', GarnetParam = GarnetParam)   
     
elif ProblemType == 'FrozenLake':
    
    # Problem-specific parameters
    state_size = 16
    
    # Define the policy
    pi = None
    # pi = 1
    # pi = 1 # np.random.randint(0, 2, self.n_states)
    # pi = 2 # [0] * self.n_states
    
    # FrozenLake problem
    MDP = mdp.FrozenLakeMDP(map_name = '4x4', is_slippery = True, render_mode = 'human', policy = pi)
    # pi = MDP.pi
    if pi is not None: # You need to know a deterministic policy beforehand
        print(f"The optimal value function obtained in matrix form is: {MDP.V}")

# Call the function
mdp.experiment_sample_behaviour_gain_adaptation(MDP, discount, pi, acc_param_list = gain_list,
                            iter_no = iter_no,
                            error_norm_ord = error_norm_ord,
                            shown_states = 'adaptive', #[10,40],
                            meta_lr = meta_lr, 
                            normalization_eps = normalization_eps,
                            normalization_flag = 'BE2',
                            gain_list = gain_list)

show()
