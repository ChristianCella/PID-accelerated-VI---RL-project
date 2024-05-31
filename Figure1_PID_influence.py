""" 
This script allows to obatin the figures 1a and 1b of the paper, depending on the policy that is specified. In this script, you can also import
the custom class 'FrozenLake', useful in the reproducibility report.
With this script it is possible to plot the norm of the error for the different gains as a function of the number of iterations.
"""

# Import standard libraries
from collections import namedtuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import *

# Import the package conatining all funcrions and classes to be used
import MDPs as mdp

# Define the main parameters of the problem
discount = 0.99
iter_no = 2000
Gain = namedtuple('Gain', ['kP', 'kI', 'kD', 'I_alpha', 'I_beta'], defaults = [1, 0, 0, 0, 0])
error_norm_ord = np.inf # or 2 or 1 or np.inf
discount_str = str(discount).replace('.','p')

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
    # pi = 1 # np.random.randint(0, 2, self.n_states)
    # pi = 2 # [0] * self.n_states
    
    # FrozenLake problem
    MDP = mdp.FrozenLakeMDP(map_name = '4x4', is_slippery = True, render_mode = 'human', policy = pi)
    
    if pi is not None: # You need to know a deterministic policy beforehand
        print(f"The optimal value function obtained in matrix form is: {MDP.V}")


# Select the gains as a function of the policy
if pi is not None:
            gain_list = [
                        Gain(1.2,0,0,0.05,0.95),
                        Gain(1,-0.4,0,0.05,0.95),
                        Gain(1,0,0.15,0.05,0.95),
                        # Gain(k_p_analyt, 0, k_d_analyt, 0.05, 0.95)
                        # Gain(1,-0.4,0.15,0.05,0.95),
                        ]

        # Random Walk (control: remember, in this case you do not specify the policy)
if pi is None:
    gain_list = [
                Gain(1.2,0,0,0.05,0.95),
                Gain(1,0.75,0,0.05,0.95),
                Gain(1,0,0.4,0.05,0.95),
                Gain(1,0.75,0.4,0.05,0.95),
                Gain(1.,0.7,0.2,0.05,0.95),
                ]

# Call the function defined above
mdp.experiment_sample_behaviour(MDP, discount, pi, acc_param_list = gain_list,
                            iter_no = iter_no,
                            error_norm_ord = error_norm_ord,
                            shown_states = 'adaptive', #[10,40],
                            gain_list = gain_list)
show()