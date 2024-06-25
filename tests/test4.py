"""
This script allows to obatin the figures 4 of the paper, depending on the policy that is specified.
NOTE: in the second experimetn described in the paper, more specifically for the Figures 4a and 4b, only garnet is considered:
as a consequnece, in this script you cannot import 'FrozenLake'. 
If you wanted to do so, you would need to change the function 'experiment_gain_adaptation_garnet'
"""
# It is not true that FL is not considered. I'm using inside the function 'experiment_gain_adaptation_garnet' defining MDP as a FL Problem'

# Import standard libraries
import numpy as np
from matplotlib.pyplot import *
from collections import namedtuple
np.random.seed(1)

# Libraries for parallelization: these are very useful when you want to run multiple experiments
from joblib import Parallel, delayed
import multiprocessing

import sys 
sys.path.append('.')
# Import all the functions defined inside 'ValueIteration.py' and import also the class defined in 'FiniteMDP.py'
import MDPs as mdp

# Specify the problem
ProblemType = 'garnet'
Gain = namedtuple('Gain', ['kP', 'kI', 'kD', 'I_alpha', 'I_beta'], defaults = [1, 0, 0, 0, 0])

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
     
# Define some constants
discount = 0.85 # 0.99 standard case for the first figure, 0.85 for the second
iter_no = 3000
runs_no = 20
error_norm_ord = np.inf # or 2 or 1 or np.inf
normalization_flag = 'BE2'
discount_str = str(discount).replace('.','p')
param_range = (0,1700) # (0,3500) for figure with gamma = 0.99, (0,1700) for gamma = 0.85

# Define the hyperparameters (gains, set of eta and epsilon)
gain = Gain(1.0, 0, 0, 0.05, 0.95)
normal_eps = 1e-20
eps_str = str(normal_eps).replace('.','p')
hyper_param_list = [(0.001, normal_eps), (0.005, normal_eps), (0.01, normal_eps),
                    (0.02, normal_eps), (0.05, normal_eps), (0.1, normal_eps),
                    ]
sweep_param_detail = 'eps='+eps_str

# Call the function defined above to see the effect of eta and epsilon on the normalized error
normalized_error = mdp.experiment_gain_adaptation_garnet(discount = discount, pi = pi, hyper_param_list = hyper_param_list,
                                state_size = state_size, action_size = action_size,
                                GarnetParam = GarnetParam,
                                runs_no = runs_no,
                                init_gain = Gain(1, 0, 0, 0.05, 0.95),
                                with_hp_model_selection = True,
                                normalization_flag = normalization_flag,
                                iter_no = iter_no,
                                error_norm_ord = error_norm_ord, param_range = param_range,)

show()


