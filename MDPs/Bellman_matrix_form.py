""" 
This simple script allows to compute the value of a policy pi (computing V_pi) in a finite MDP using the matrix form of the Bellman expectation equation.
NOTE: when considering MDPs, actions must be accounted for by introducing a policy pi that specifies the action to be taken in each state. based on this,
all the quantitires appearing in the m,atrix form of the Bellman Expectation Equation need to have the subscript 'pi'.
The Bellman Equation in matrix form is:
    V_pi = inv(I - gamma * P_pi) * R_pi
The 
where:
    * V_pi is the value function of the policy pi
    * I is the identity matrix
    * gamma is the discount factor
    * P_pi is the transition matrix of the MDP under the policy pi (P is a tensor (states x states x actions), not a matrix; P_pi is a matrix)
    * pi is the policy
"""

from finiteMDPs import * # Import the class defined in another module
from numpy.linalg import inv
import numpy as np

# Set some parameters
np.random.seed(1)
state_size = 50   
discount = 0.99
iter_no = 2000

verbose = True

# Create an instance of class FiniteMDP
MDP = FiniteMDP(state_size, ActionSize = 4, ProblemType = 'randomwalk', GarnetParam = (3, 5))

# Get the transition matrix and the reward matrix: P is actualy a tensor, not a matrix
P = np.array(MDP.P)
R = np.array(MDP.R)

# Specify a deterministic policy 
pi = [0] * state_size # always play the first action in each state
# pi = np.random.randint(0, 2, state_size)
# pi = None # This is the greedy policy ==> Control case (in control I want to evaluate the optimnal policy associated to V*)

if verbose:
    print(f"The policy is {pi}, R is {R} and the tensor P is {P}")

# Initialize the matrix P_pi: you want to pass from a tensor P to a matrix, whose creation is driven by the policy pi   
P_pi = np.zeros((state_size, state_size))
R_pi = np.zeros((state_size, 1))

for s in range(state_size):
    a = pi[s]
    P_pi[s] = P[a, s]
    # R_pi[s] = np.dot(a, R[s])
    R_pi[s] = R[s]

# Compute the value of the policy pi (Bellman expectation equation in matrix form)  
I = np.eye(state_size)
V = np.dot(inv(I - discount * P_pi), R_pi)

if verbose:
    print(f"The matrix P_pi is: {P_pi}")
    print(f"The matrix R_pi is: {R_pi}")
    print(f"The optimal value of V calculated with the matrix form is: {V}")