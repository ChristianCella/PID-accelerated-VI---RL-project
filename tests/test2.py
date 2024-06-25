""" 
This script allows to obatin the figures 2 of the paper, depending on the policy that is specified.
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
iter_no = 500
Gain = namedtuple('Gain', ['kP', 'kI', 'kD', 'I_alpha', 'I_beta'], defaults = [1, 0, 0, 0, 0])
rlocus_flag = False # To plot also additional information about the eigenvalues using the root locus
error_norm_ord = np.inf # or 2 or 1 or np.inf
discount_str = str(discount).replace('.','p')
param_name = 'kD'
param_range = (0.5,0.9)

resolution = 100 # resolution of the controller gain study

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
    
    # pi = MDP.pi # Get the policy from the MDP class
    
    if pi is not None: # You need to know a deterministic policy beforehand
        print(f"The optimal value function obtained in matrix form is: {MDP.V}")

# Define the default gain
gain_default = Gain(1., 0.0, 0., 0.05, 0.95)

# Call the function
mdp.experiment_1D_param_sweep(MDP, discount, pi, 
                        param_name = param_name,
                        param_range = param_range, resolution = resolution,
                        gain_default = gain_default,
                        iter_no = iter_no,
                        error_norm_ord = error_norm_ord) # Note that this is for the sup-norm

# Additional check if you also want to plot the root locus
if rlocus_flag:
    file_name_detail = ProblemType +'(discount='+discount_str+')-root locus-'+ param_name +\
                        ('(control)' if pi is None else '(PE)')

    # NOTE: the root locus doesn't make sense for the control case
    if pi is None:
        print("Warning! Root locus doesn't make sense for the control case.")
    
    # Set a dictionary of functions returning the error dynamics
    P_controller = {'kP': lambda k: mdp.P_with_P(P, discount, k_p=k),
                    'kI': lambda ki: mdp.P_with_PI(P, discount, ki, beta = gain_default.I_beta, alpha=gain_default.I_alpha),
                    'kD': lambda kd: mdp.P_with_PD(P, discount, kd, D_term_type='basic')
                    }

    P = np.matrix(MDP.P[0])
    P_fn = P_controller[param_name]
    
    # call the function
    P_eig = mdp.plot_roots(P_fn, param_range = param_range,
                        fig_filename = file_name_detail)
    
show()