"""
Created on Thu Mar  7 16:47:49 2019
@author: Amir-massoud Farahmand
----------------------------------

This file conatins the definitiojn of the Markov Decision Processes (MDPs) that are used in the RL algorithms.
It must be pointed put that the state-transition matrix is in reality a tensor state x state x action. If you want to use the matrix form
to solve Bellman Expectation equation, you have to pass to P_pi, which is a matrix state x state. This is done by choosing a policy pi.
"""

"""
Define the class for the MDPs used in the paper 
"""

import numpy as np
import random, array

class FiniteMDP:

    # Type constructor
    def __init__(self, StateSize = 5, ActionSize = 2, ProblemType = None,
                 GarnetParam = (1,1), TwoThreeStatesParam = None):
        
        self.StatesNo = StateSize
        self.ActionsNo = ActionSize
        self.P = [np.matrix(np.zeros( (StateSize,StateSize))) for act in range(ActionSize)]       
        self.R = np.zeros( (StateSize,1) )
        
        # define some problem-specific variables
        if ProblemType == 'garnet':
            b_P = GarnetParam[0] # branching factor
            b_R = GarnetParam[1] # number of non-zero rewards
                    
        if ProblemType == 'TwoThreeStatesProblem':
            
            if TwoThreeStatesParam is None:
                TwoThreeStatesParam = 0.0

            self._TwoThreeStateProblem(p = TwoThreeStatesParam)
            return
        
        """ 
        Create the state-transition matrix P
        """
        for act in range(ActionSize):
            for ind in range(StateSize):
                pVec = np.zeros(StateSize)
                
                
                if ProblemType == 'garnet':
                    
                    # Garnet-like (not exact implementaiton).
                    p_vec = np.append(np.random.uniform(0,1,b_P - 1),[0,1])
                    p_vec = np.diff(np.sort(p_vec))
                    pVec[np.random.choice(StateSize,b_P, replace = False)] = p_vec

                elif ProblemType == 'randomwalk':
                    if act == 0:
                        pVec[ (ind + 1) % StateSize ] = 0.7 # Walking to the right!
                        pVec[ ind ] = 0.2
                        pVec[ (ind - 1) % StateSize ] = 0.1
                    else:
                        pVec[ (ind - 1) % StateSize ] = 0.7 # Walking to the left!
                        pVec[ ind ] = 0.2                        
                        pVec[ (ind + 1) % StateSize ] = 0.1
                        
                # NOTE: you'd better not use the following code. It's not correct.
                elif ProblemType == 'smoothrandomwalk':
                    if act == 0:
                        pVec[ min(ind + 1,StateSize):min(ind+5,StateSize) ] = 0.7/5 # Walking to the right!
                        pVec[ ind ] = 0.2/5
                        pVec[ max(ind - 5,0): ind] = 0.1/5
                    else:
                        pVec[ max(ind - 5,0): ind] = 0.7/5 # Walking to the left!
                        pVec[ ind ] = 0.2/5
                        pVec[ min(ind + 1,StateSize):min(ind+5,StateSize) ] = 0.1/5                        
                       
                elif ProblemType is None:
                    pVec = np.random.exponential(1,StateSize)
                
                # Obtain the final tensor
                pVec /= sum(pVec)
                self.P[act][ind,:] = pVec
            
        """ 
        Create the reward vector R
        """
        if ProblemType == 'garnet':
            self.R[np.random.choice(StateSize,b_R, replace = False)] = np.random.uniform(0,1,b_R)[:,np.newaxis]
        elif ProblemType == 'randomwalk':
            self.R[10] = -1.
            self.R[-10] = 1.
        elif ProblemType == 'smoothrandomwalk':
            self.R[0:3] = 1.
            self.R[-3:] = 1.
            self.R[int(StateSize*0.45):int(StateSize*0.55)] = 1.1            
        else:
            self.R = np.random.uniform(0,1,StateSize)     

    """ 
    The following class-specific methods are not used to obtain the results reported in the paper. With these, you can obtain the 
    results presented in the Appendix though.
    """
    def _TwoThreeStateProblem(self, p = 0):

        if self.StatesNo == 2:
            # Two real eigenvalues
            if p > 1 or p < 0:
                print("(FiniteMDP) Out of range parameter for TwoStateReal Problem.")
                p = max(0,min(p,1))

            P = [[1-p, p],
                 [p, 1-p]]
        elif self.StatesNo == 3:
            if p > 1/3 or p < 0:
                print(p)
                print("(FiniteMDP) Out of range parameter for ThreeStateComplex Problem.")
                p = max(0,min(p,1/3))
                
            # One real eigenvalue (1) and two complex conjugate with zero real component
            P = [[1/3, 1/3+p, 1/3-p],
                 [1/3-p, 1/3, 1/3+p],
                 [1/3+p, 1/3-p, 1/3]]

        P = np.matrix(P)
        P = P/np.sum(P,axis = 1)

        for act in range(self.ActionsNo):
            self.P[act] = P

        self.R[0] = 1
        self.R[self.StatesNo-1] = -0.98

        
    def DrawSamplesFromState(self,SamplesNo = 1, x0 = 0, a0 = 0):
        
        p = array(self.P[a0][x0,:])
        p = p[0,:]
        X = random.multinomial(SamplesNo,p,size=1)
        return X
        
        
    def DrawSamples(self,SamplesNo = 1,NextStateSamplesNo = 1, act = 0):
        
        X = []
        XNext = []
         
        for ind in range(SamplesNo):
            x0 = random.randint(0,self.StatesNo,1)
            xNext = self.DrawSamplesFromState(NextStateSamplesNo,x0, a0 = act)
            
            X.append(x0)
            XNext.append(xNext)
            
        return X,XNext

""" 
Define the class for the FrozenLake MDP
"""

import gymnasium as gym
from numpy.linalg import inv

class FrozenLakeMDP:
    
    def __init__(self, map_name = '8x8', is_slippery = True, render_mode = 'human', policy = None):
        
        self.env = gym.make('FrozenLake-v1', map_name = map_name, render_mode = render_mode, is_slippery = is_slippery) 
        self.n_states = self.env.observation_space.n
        self.n_actions = self.env.action_space.n
        self.P_mat = self.env.P # Very complicated list of lists
        self.R = np.zeros((self.n_states, 1)) # Specific for the FrozenLake problem
        self.R[-1] = 1 # Only the last state has a reward of 1
        
        # verify if a policy is needed       
        if policy == 1:
            
            self.pi = np.random.randint(0, 2, self.n_states)
            
        elif policy == 2:
            
            self.pi = [0] * self.n_states
        
        # calculate the P tensor
        P = [np.matrix(np.zeros( (self.n_states,self.n_states))) for act in range(self.n_actions)]
        
        for state in range(self.n_states):
            for action in range(self.n_actions):
                transitions = self.P_mat[state][action]
                for prob, next_state, reward, done in transitions:
                    P[action][state, next_state] += prob
        """ 
        for action in range(self.n_actions):
            for state in range(self.n_states):
                transitions = self.P_mat[state][action]
                for prob, next_state, reward, done in transitions:
                    P[action][state, next_state] += prob
        """
        
        """
        P = np.zeros((self.n_actions, self.n_states, self.n_states))
        for action in range(self.n_actions):
            for state in range(self.n_states):
                transitions = self.P_mat[state][action]
                for prob, next_state, reward, done in transitions:
                    P[action, state, next_state] += prob
        """
        # This is the tensor P defining the transition probabilities   
        self.P = P
        print(f"Tensor P: {self.P}")        
        # self.P = np.matrix(P).reshape(self.n_actions, self.n_states, self.n_states) # actions x states x states
        
        # In case that a policy is reqired, calculate the value of the policy implementing Bellman expectation in matrix form (Prediction)
        if policy is not None:
            
            P_pi = np.zeros((self.n_states, self.n_states))

            for s in range(self.n_states):
                a = self.pi[s]
                P_pi[s] = P[a][s]
            """ 
            for s in range(self.n_states):
                a = self.pi[s]
                P_pi[s] = P[a, s]
            """

            # Compute the value of the policy pi (Bellman expectation equation in matrix form)  
            I = np.eye(self.n_states)
            self.V = np.dot(inv(I - 0.99 * P_pi), self.R)
            
                          
""" 
Simple test of the classes
"""   
if __name__ == '__main__':

    from matplotlib.pyplot import *
    
    matplotlib.rcParams['mathtext.fontset'] = 'cm' # 'cm' or 'stix'
    matplotlib.rcParams['font.family'] = 'STIXGeneral'

    matplotlib.rc('xtick', labelsize = 15)
    matplotlib.rc('ytick', labelsize = 15)
    
    # Create an instance of the class defined above
    # Problem = FiniteMDP(25, ProblemType = 'garnet', GarnetParam = (5,2) )
    Problem = FrozenLakeMDP(map_name = '8x8', is_slippery = True, render_mode = 'human', policy = 2)
    
    # Display the state transition matrix P (if a policy was specified, only for the case of FrozenLake)
    # print(f"The optimal value of V calculated with the matrix form is: {Problem.V}")

    
    # Display the matrix P, for a certain action, as an image
    imshow(Problem.P[0], interpolation = 'None')
    colorbar()
    xlabel('Target state', fontsize = 15)
    ylabel('Starting state', fontsize = 15)
    title(f'State transition matrix P - action {0}', fontsize = 20)
    
    # Plot the trend of the reward
    figure()
    plot(Problem.R)
    xlabel('N states', fontsize = 15)
    ylabel('R (not $R^{\pi}$)', fontsize = 15)
    title(f'Trend of the reward', fontsize = 20)
    grid()
    show()


