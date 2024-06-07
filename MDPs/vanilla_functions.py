"""
Created on Thu Mar  7 16:46:48 2019
@author: Amir-massoud Farahmand

Explanation of the code
-----------------------

This code contains all the 'vanilla' functions that are used by some more complex functions defined in 'tests_functions'.
In the test at the end of the code  (if __name__== 'main'), some classes are imported and a small test can be performed.
    
The paper aims at solving a problem under two different perspectives: Prediction and Control, leveraging Value Iteration. Of course, in the case
presented in the paper, the transition model P is assumed to be known: Dynamic Programming case. 
Considering the discrete settings of the problem, the equations can be written as follows (in case of Prediction, I can leverage the matrix form):
    * Prediction (I just want to evaluate V): V = R + gamma * P * V ==> V = (I - gamma * P) * R ==> This is the Bellman EXPECTATION Equation.
    * Control (I want to evaluate V and get the policy): V = max_a [R + gamma * P * V] ==> This is the Bellman OPTIMALITY Equation.
        After I know V*, I can compute the optimal policy ==> pi*(s) = argmax_a [R(s) + gamma * sum_s' [P(s'|s,a) * V*(s')]]
"""

import numpy as np
import matplotlib
from matplotlib.pyplot import *
from collections import namedtuple

matplotlib.rcParams['mathtext.fontset'] = 'cm' # 'cm' or 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

matplotlib.rc('xtick', labelsize = 15)
matplotlib.rc('ytick', labelsize = 15)

Gain = namedtuple('Gain', ['kP', 'kI', 'kD', 'I_alpha', 'I_beta'], defaults = [1, 0, 0, 0, 0])

# Function to compute the value function in the non.accelerated case
def value_iteration(R, P, discount, IterationsNo = None, policy = None):
    
    """
    Value Iteration for finite state and action spaces.
    
    NOTE: if you only needed to stay in a 'Prediction' case, you could already evaluate the Value function in matrix form.
    For a better explanation, see the script called 'VI_matrix_form.py': you suppose to have a certain policy and, once the MDP
    is selected, P and R are known. The problem is that P is a tensor (actions x states x states) and R is a matrix (states x 1).
    This is the meaning of P_pi: it is a matrix (states x states) that contains the transition probabilities dtermined by the policy:
    basically, you are selecting the rows of P that correspond to the actions specified by the policy.
    
    NOTE:
    Here, there is no need to have a greedy improvement ('policy_greedy' function), because we do not want to to calculate the policy:
    despite the possibility to calculate the optimal value function also for Control (Q), we do not want to calculate the optimal policy
    associated to it.
    """    
    action_size, state_size = np.shape(P)[0:2] # states = 50, actions = 4
    print(f"The number of actions is: {action_size} and the number of states is: {state_size}")
    
    R = R.reshape(state_size,1) # reshape the rewards as an array 50x1       
    V = np.zeros( (state_size,1) ) # Initialize the value function to zero
    Q = np.matrix(np.zeros( (state_size, action_size) ) ) # Initialize the Q-values to zero
    
    # in case the number of iterations is not provided, set a default value based on the discount factor
    if IterationsNo == None:
        IterationsNo = int(10 / (1 - discount))
        
    # Debugging
    VTrace = np.zeros( (state_size,IterationsNo)) # each column stores a value function for a specific iteration
    
    # Initialize the value function to zero (this line of code is completely useless)
    V = np.max(Q, axis = 1).reshape(state_size, 1)
    
    # At this point, in Prediction, you would already know how to calculate V in matrix form
    
    # start the iterations
    for iter in range(IterationsNo):
        # Both the Bellman expectation operator (prediction) and the Bellman optimality operator (control) require to start from this term.
        for act in range(action_size): # Compile column by column (4 columns in total)
            Q[:, act] = R + discount * P[act] * V # 50x4 matrix
    
        """ 
        Compute the optimal Value function based on te case you chose:
            * Control ==> no policy is specified becasue you want to comput it after the value function is computed.
                ° The optimal value function is obtained by maximizing the term calculated above.
            * Prediction ==> the policy is specified because you want to evaluate the value function based on the policy.
                ° The optimal value function is obtained by implementing this formula: V_pi = inv(I - gamma * P_pi) * R_pi
        """
        if policy is None: # Control case
            V = np.max(Q, axis = 1).reshape(state_size,1) # Reshape into a 50x1 column vector
        else: # Prediction case (the result is the same as the one you would obtain by using the formula in 'VI_matrix_form.py')
            V = np.array([Q[m, policy[m]] for m in range(state_size)]).reshape(state_size, 1)
        # Append the 'iter-th' value function
        VTrace[:, iter] = V.reshape( (state_size,) )
    
    return V, Q, VTrace    

# Accelerated Value Iteration for finite state and action MDP ==> this is the version used in the paper
def value_iteration_with_acceleration_new(R, P, discount, IterationsNo = None, policy = None, alpha = 0.0,
                                            accelation_type = None, gain = Gain(1., 0, 0),
                                            gain_adaptation = False, meta_lr = None,
                                            normalization_flag = 'BE2', normalization_eps = 1e-6):
    
    """
    Value Iteration for finite state and action spaces. It supports accelerated variants, such as the PID one. This is used for the paper.
    Basically, this function implements the set of equations below (for Predictiuon; for Control they are the same except that you use Q and not V):
        * z_k+1 = beta * z_k + alpha * BR(V_k)
        * V_k+1 = (1 - Kp) * Vk + Kp * T_pi * V_k + Ki * z_k+1 + Kd * (V_k - V_k-1)
    With:
        * alhpa and beta chosen so that their sum is equal to 1
        * BR(V_k) = T_pi * V_k - V_k
    In the code:
        * dQ = V_k - V_k-1 (or Qk - Qk-1)
        * K_d_mat = Kd
        * BE_Q = BR(V_k) or BR(Q_k)
        * BE_Q_integ = z_k+1
        * V_tmp (or Q_tmp) = Vk (or Qk)
        * Q_new (or V_new) = T_pi * Vk (or T* * Qk) ==> This is the Bellman operator
    """    
    
    action_size, state_size = np.shape(P)[0:2]
    
    R = R.reshape(state_size,1)        
    V = np.zeros( (state_size,1) )
    Q = np.matrix(np. zeros( (state_size,action_size) ) )
        
    # Simple heuristic to chooce the iteration number, if not specified.
    if IterationsNo == None:
        IterationsNo = int(10 / (1 - discount))

    # Construct an approximate Phat (this is used in the Ishikawa acceleration type)
    Phat = []
    
    for a in range(action_size):
        # p_hat = np.matrix( np.diag(np.diag(P[a])) )
        p_hat = P[a]*(np.eye(state_size) + 0.2*np.random.rand(state_size, state_size) )
        p_hat = p_hat/np.sum(p_hat,axis = 1)
        Phat.append( p_hat)

    # Create variables to keep track of the values of variables as the iterations proceed
    V_trace = np.zeros( (state_size,IterationsNo))
    Q_trace = []
    pi_trace = []

    # initialize the value function to zero
    V = np.max(Q,axis = 1).reshape(state_size,1)

    # intiialize all the terms needed in the final computation
    dQ = 0 * Q
    dQ_trace = []
    
    dV = 0*V
    dV_trace = []
    
    z_trace = []
    
    # Controller
    K_d_mat = np.zeros_like(P) # Matrix gain (scalar will be multiplied)
    
    for act in range(action_size):
        K_d_mat[act] = (0.0*P[act] + 1.0 * np.eye(state_size)) # ther act-th component of the gain matrix is an identity

    # Gain adaptation (for how the code works now, this is useless)
    lam = 1.

    BE_Q = 0*Q
    BE_V = 0*V
    beta = 0.9 # Parameter for BE averaging

    BE_Q_trace = []
    # BE_Q_imm_trace = []

    # These are the states of the integrator of PID
    BE_Q_integ = 0*BE_Q
    BE_V_integ = 0*BE_V

    BE_Q_integ_trace = []

    # Initialize the variables to keep track of all the gains
    kD_trace = []
    kI_trace = []
    kP_trace = []

    kP_grad_trace = []
    kI_grad_trace = []
    kD_grad_trace = []
    gain_trace = []

    # Loop through all the iterations
    for iter in range(IterationsNo):
        
        # Check if the iteration number is a multiple of 50
        if np.mod(iter,50) == -1:
            print(iter, alpha)

        # Create independent copies of Q and V
        Q_tmp = np.copy(Q)
        V_tmp = np.copy(V)

        # Account for different possibilities of acceleration (in the paper only PID is used)
        if accelation_type in {'P', 'PD', 'PI', 'PID'}:
            V_new, Q_new = Bellman_operator(R, P, discount, V, policy = policy) # Call the function to calculate the Bellman operator           
        if accelation_type == 'Picard' or accelation_type == None:

            # Picard Iteration
            # V, Q = Bellman_operator(R, P, discount, V, policy = policy)
            V_new, Q_new = Bellman_operator(R, P, discount, V, policy = policy)

        if accelation_type == 'IntertialMann':
            # Inertial Mann Iteration
            z = V + alpha*dV
            V_new, Q_new = Bellman_operator(R, P, discount, z, policy = policy)

            V_tmp_Tinstad, Q_tmp_Tinstad = Bellman_operator(R, P, discount, V, policy = policy)

            # V_new, Q_new = Bellman_operator(R, P, discount, V + alpha*dV, policy = policy)
            z_trace.append(V_tmp_Tinstad - z)

        if accelation_type == 'IntertialMannWithBE':
            # Inertial Mann Iteration
            z = V + alpha*BE_V + 0.1*dV
            V_new, Q_new = Bellman_operator(R, P, discount, z, policy = policy)
            # V_new, Q_new = Bellman_operator(R, P, discount, V + alpha*dV, policy = policy)
            # z_trace.append(BE_V)
            V_tmp_Tinstad, Q_tmp_Tinstad = Bellman_operator(R, P, discount, V, policy = policy)
            z_trace.append(V_tmp_Tinstad - z)

        if accelation_type == 'Ishikawa':
            # z = V + alpha*dV
            V_tmp, Q_tmp = Bellman_operator(R, Phat, discount, V, policy = policy)
            # z = (1 - alpha)*V + alpha*V_tmp
            z = (1 - alpha)*V + alpha/2*(V_tmp + dV) # Q: Why 1/2? And is this a Derivative version?
            V_new, Q_new = Bellman_operator(R, P, discount, z, policy = policy)

        # Possibility to weigh the new and the old Q (lam is usually 1)
        Q = (1-lam)*Q + lam*Q_new

        # Compute the Bellman Residual and the variable z_k+1
        BE_Q = Q_new - Q_tmp # Bellman error
        BE_Q_integ = gain.I_beta * BE_Q_integ + gain.I_alpha * BE_Q # z_k+1

        # Tracing of the Bellman Error and its temporally averaged one (The variable called z_k+1 in the paper)
        BE_Q_trace.append(BE_Q)
        BE_Q_integ_trace.append(BE_Q_integ)

        # Compute the Bellman Residual for the V function (Prediction case); In this case, no 'append' follows
        BE_V = V_new - V_tmp
        BE_V_integ = gain.I_beta*BE_V_integ + gain.I_alpha*BE_V

        # Initialize the last term of V_k+1 (or Q_k+1)
        dQ_corr = 0 * dQ
        
        # Obtain all the possible derivatives of the Bellman Residual
        for act in range(action_size):
            
            # Compute the derivative term of the controller (K_d_mat is an identity matrix, not a scalar)
            dQ_corr[:, act] = K_d_mat[act] * dQ[:, act]
        
        # Define the V (or Q, in case of Control) function according to the type of controller you need
        if accelation_type == 'P':

            Q = (1 - gain.kP) * Q_tmp + gain.kP * Q_new

        if accelation_type == 'PD':

            Q = (1 - gain.kP) * Q_tmp + gain.kP * Q_new + gain.kD * dQ_corr 

        if accelation_type == 'PID':

            Q = (1 - gain.kP)*Q_tmp + gain.kP*Q_new + gain.kD*dQ_corr + gain.kI*BE_Q_integ 


        # Update the difference (for Control)
        dQ = Q - Q_tmp
        
        # Calculate the value function according to Prediction or Control (look at the functrion value_iteration defined above) 
        if policy is None:
            V = np.max(Q,axis = 1).reshape(state_size,1)
        else:
            V = np.array([Q[m,policy[m] ] for m in range(state_size)]).reshape(state_size,1)

        # Update the difference (for Prediction)
        dV = V - V_tmp

        """ 
        Start of the gain adaptation procedure.
        """
        if gain_adaptation and iter > 1: 
            
            # Access the time histories and get the previous values
            V_km1 = V_trace[:,iter - 1].reshape( (state_size,1) )
            V_km2 = V_trace[:,iter - 2].reshape( (state_size,1) )

            Q_km1 = Q_trace[iter - 1]
            Q_km2 = Q_trace[iter - 2]
            
            # Compute the gradient of the Bellman Error w.r.t. the PID gains
            kD_grad, kI_grad, kP_grad = adapt_gain_new(R, P, discount, policy,
                                                    Q_k = Q, V_k = V,
                                                    Q_km1 = Q_km1, V_km1 = V_km1,
                                                    Q_km2 = Q_km2, V_km2 = V_km2,
                                                    z_k = BE_Q_integ,
                                                    normalization_flag = normalization_flag,
                                                    normalization_eps = normalization_eps)
            
            # Update the gains, following the negative direction of the gradient
            kD_new = gain.kD - meta_lr * kD_grad
            kI_new = gain.kI - meta_lr * kI_grad
            kP_new = gain.kP - meta_lr * kP_grad
            
            # Update the gains
            gain = Gain(kP_new, kI_new, kD_new, gain.I_alpha, gain.I_beta)
            
            # display some information if the iteration is multuiple of 50
            if iter%50 == -1:
                print(iter, gain)

            # Augment all the variables concerning gains
            kD_trace.append(kD_new)
            kI_trace.append(kI_new)
            kP_trace.append(kP_new)

            gain_trace.append(gain)

            kP_grad_trace.append(kP_grad)
            kI_grad_trace.append(kI_grad)
            kD_grad_trace.append(kD_grad)

        # Tracing (regardless of the fact that the gain adaptation is active or not)
        dV_trace.append(dV)
        dQ_trace.append(dQ)
        Q_trace.append(np.copy(Q))
        V_trace[:,iter] = V.reshape( (state_size,) )
        
        # Policy improvement step ==> Update the policy in case of control: choose the greedy update. this is the optimal policy at a certain iteration
        pi_trace.append(policy_greedy(Q)) # pi_trace will be a matrix of 50xn_iters
        
    # Transform all the lists into numpy arrays
    dQ_trace = np.array(dQ_trace)
    dV_trace = np.array(dV_trace).squeeze()
    Q_trace = np.array(Q_trace)
    pi_trace = np.array(pi_trace)    
    z_trace = np.array(z_trace).squeeze()
    BE_Q_trace = np.array(BE_Q_trace)
    BE_Q_integ_trace = np.array(BE_Q_integ_trace)

    # Possible plot
    if gain_adaptation:
        figure(figsize = (10, 6))
        
        # Display the gain trend
        subplot(2,1,1)
        plot(kP_trace,'b', linewidth = 2)
        plot(kI_trace,'r', linewidth = 2)        
        plot(kD_trace,'k', linewidth = 2)
        legend(['$k_p$', '$k_I$', '$k_d$'], fontsize = 15)
        xlabel('Iteration', fontsize = 20)
        ylabel('Controller gains', fontsize = 20)
        title('variation of the gains', fontsize = 20)
        grid()

        # Display the logaritmic derivatives of the gains
        subplot(2,1,2)
        plot(np.log(np.abs(kP_grad_trace)),'b')
        plot(np.log(np.abs(kI_grad_trace)),'r')
        plot(np.log(np.abs(kD_grad_trace)),'k')

        legend(['$k_p$', '$k_I$', '$k_d$'], fontsize = 15)
        xlabel('Iteration', fontsize = 20)
        ylabel('Log of gain derivatives', fontsize = 20)
        
        grid()
   
    return V, Q, V_trace, Q_trace, dV_trace, dQ_trace, z_trace, BE_Q_trace, BE_Q_integ_trace, gain_trace


# function that returns the new value function after applying the Bellman expectation (Prediction) or Bellman optimality (Control)
def Bellman_operator(R,P,discount, V = None, Q = None, policy = None):
    
    """
    Bellman operator for finite state and action spaces.
    """    
            
    action_size, state_size = np.shape(P)[0:2] # 4 actions, 50 states
    R = R.reshape(state_size,1) # 50x1      
    Q_new = np.matrix(np.zeros( (state_size,action_size) ) ) # Initialize a 50x4 matrix full of zeros
    
    # Compute the value function if not given (same as described in the function 'value_iteration')
    if V is None:
        if policy is None: # Control
            V = np.max(Q,axis = 1).reshape(state_size,1)
        else: # Prediction
            V = np.array([Q[m,policy[m] ] for m in range(state_size)]).reshape(state_size,1)

    # Both the Bellman expectation operator (prediction) and the Bellman optimality operator (control) require to start from this term.
    for act in range(action_size):
        Q_new[:, act] = R + discount *P[act] * V
    
    # After applying Bellman to obtain the common term, calculate the new term as a function of Control or Prediction
    if policy is None:
        V_new = np.max(Q_new,axis = 1).reshape(state_size, 1)
    else:
        V_new = np.array([Q_new[m, policy[m] ] for m in range(state_size)]).reshape(state_size, 1)

    return V_new, Q_new

# Function to return the greedy policy for an action-value function Q (for Control)
def policy_greedy(Q, x = None):

    if x is None:
        x = range(np.shape(Q)[0]) # This number is equal to the number of states (50)
    act = np.argmax(Q[x, :], axis = 1) # for all the x lines, go through all the columns and find the index corresponding to the max of Q 

    return np.array(act).squeeze() # Return the greedy policy that is a vector of 50 elements

# Calculate all the gradients
def adapt_gain_new(R, P, discount, policy, Q_k, V_k, Q_km1, V_km1, Q_km2, V_km2, z_k = None,
                   normalization_flag = True, normalization_eps = 1e-8,
                   truncation_flag = True, truncation_threshold = 1.):
    
    """
    Gain adaptation for PID gains.
    It is based on gradient descent on the Bellman Error.
    This is used for the paper.
    """   

    if policy is None: # case of control: Choose the greedy policy
        policy_k = policy_greedy(Q_k)

    # BE(V_k) is needed for all terms, so let's compute TV_k and TQ_k first
    TV_k, TQ_k = Bellman_operator(R, P, discount, V = V_k, policy = policy)

    # pass to array
    TV_k = np.array(TV_k)
    TQ_k = np.array(TQ_k)
    
    # Needed for the kD term
    BE_V_k = TV_k - V_k 
    BE_Q_k = TQ_k - Q_k

    if policy is None:
        BE_k = BE_Q_k
    else:
        BE_k = BE_V_k

    """ 
    BE(V_{k-1}) is needed for the kP term, so you need to compute T V_{k-1}.
    This has already been computed in the pervious iteration of VI, so re-computation isn't needed; it is done again to simplify the book keeping.
    """
    TV_km1, TQ_km1 = Bellman_operator(R, P, discount, V_km1, policy = policy)
    TV_km1 = np.array(TV_km1)
    TQ_km1 = np.array(TQ_km1)
    BE_V_km1 = TV_km1 - V_km1
    BE_Q_km1 = TQ_km1 - Q_km1

    action_size, state_size = np.shape(P)[0:2]

    ############################################################
    # Computing the gradient of BE(V_k) and BE(Q_k) w.r.t. kD  #
    ############################################################
    P_a_deltaV = np.matrix(np.zeros( (state_size,action_size) ) )
    
    if policy is None:
        deltaQ = Q_km1 - Q_km2
        deltaV = np.array([deltaQ[m,policy_k[m]] for m in range(state_size)]).reshape(state_size,1)        
    else:
        deltaV = V_km1 - V_km2
    
    for act in range(action_size):
        P_a_deltaV[:,act] = P[act]*deltaV
    
    if policy is None:
        grad_BE_wrt_kD = discount*P_a_deltaV - deltaQ        
    else:            
        P_deltaV = np.array([P_a_deltaV[m,policy[m] ] for m in range(state_size)]).reshape(state_size,1)
        grad_BE_wrt_kD = discount*P_deltaV - deltaV        


    if policy is None:
        grad_J_wrt_kD =  np.sum(np.multiply(BE_Q_k, grad_BE_wrt_kD))
    else:    
        grad_J_wrt_kD =  np.sum(BE_V_k * grad_BE_wrt_kD)

    ############################################################
    # Computing the gradient of BE(V_k) and BE(Q_k) w.r.t. kI  #
    ############################################################
    P_a_z = np.matrix(np.zeros( (state_size,action_size) ) )

    # z_k is of state-action dimension. This is the state-dimension version of it.
    if policy is None:
        z_k_V = np.array([z_k[m,policy_k[m] ] for m in range(state_size)]).reshape(state_size,1)
    else:
        z_k_V = np.array([z_k[m,policy[m] ] for m in range(state_size)]).reshape(state_size,1)


    for act in range(action_size):
        P_a_z[:,act] = P[act]*z_k_V
    
    if policy is None:
        grad_BE_wrt_kI = (discount*P_a_z - z_k)
    else:        
        P_z = np.array([P_a_z[m,policy[m] ] for m in range(state_size)]).reshape(state_size,1)
        grad_BE_wrt_kI = (discount*P_z - z_k_V)


    if policy is None:
        grad_J_wrt_kI =  np.sum(np.multiply(BE_Q_k, grad_BE_wrt_kI))
    else:            
        grad_J_wrt_kI =  np.sum(BE_k * grad_BE_wrt_kI)


    ################################################
    # Computing the gradient of BE(V_k) w.r.t. kP #
    ################################################

    TV_km1, TQ_km1 = Bellman_operator(R, P, discount, V_km1, policy = policy)
    TV_km1 = np.array(TV_km1)
    TQ_k = np.array(TQ_k)
    BE_V_km1 = TV_km1 - V_km1
    BE_Q_km1 = TQ_km1 - Q_km1

    if policy is None:
        # BE_km1 = BE_Q_km1
        # This is the Bellman Residual BR^*(Q_{k-1}) with action selected according to the greedy
        # policy w.r.t. Q_k. This is going to be used when we evaluate
        # P^{pi(Q_k)} BR^*(Q_{k-1})
        BE_km1 = np.array([BE_Q_km1[m,policy_k[m]] for m in range(state_size)]).reshape(state_size,1)        
    else:
        BE_km1 = BE_V_km1


    P_a_BE_km1 = np.matrix(np.zeros( (state_size,action_size) ) )
    
    for act in range(action_size):
        P_a_BE_km1[:,act] = P[act]*BE_km1
    
    if policy is None:
        grad_BE_km1_wrt_kP = discount*P_a_BE_km1 - BE_Q_km1
    else:
        P_BE_km1 = np.array([P_a_BE_km1[m,policy[m] ] for m in range(state_size)]).reshape(state_size,1)
        grad_BE_km1_wrt_kP = discount*P_BE_km1 - BE_km1

    if policy is None:
        grad_J_wrt_kP =  np.sum(np.multiply(BE_Q_k, grad_BE_km1_wrt_kP))
    else:    
        grad_J_wrt_kP =  np.sum(BE_V_k * grad_BE_km1_wrt_kP)

    # Normalization of the loss function described in the paper
    if normalization_flag:

        # Don't look at the first if; look at 'BE2'
        if normalization_flag == 'original':
            grad_J_wrt_kD /= (np.linalg.norm(BE_k)*np.linalg.norm(grad_BE_wrt_kD) + normalization_eps)
            grad_J_wrt_kI /= (np.linalg.norm(BE_k)*np.linalg.norm(grad_BE_wrt_kI) + normalization_eps)
            grad_J_wrt_kP /= (np.linalg.norm(BE_k)*np.linalg.norm(grad_BE_km1_wrt_kP) + normalization_eps)
        elif normalization_flag == 'BE2':
            # This actually works fine!
            # ORIGINAL formulation (for BE2, not the ORIGINAL above!). This is equivalent of taking the derivative of log(BE^2)
            # BE_squared = np.linalg.norm(BE_k)**2 + normalization_eps # XXX ORIGINAL XXX

            # This is what we present in the paper.
            # This is the derivative of the ratio of BE from k-1 to k.
            BE_squared = np.linalg.norm(BE_km1) ** 2 + normalization_eps
            grad_J_wrt_kD /= BE_squared
            grad_J_wrt_kI /= BE_squared
            grad_J_wrt_kP /= BE_squared
        else:
            print('(adapt_gain_new) Incorrect choice of normalization!')

    if truncation_flag:
        grad_J_wrt_kD = np.clip(grad_J_wrt_kD, -truncation_threshold, truncation_threshold)
        grad_J_wrt_kI = np.clip(grad_J_wrt_kI, -truncation_threshold, truncation_threshold)
        grad_J_wrt_kP = np.clip(grad_J_wrt_kP, -truncation_threshold, truncation_threshold)

    return grad_J_wrt_kD, grad_J_wrt_kI, grad_J_wrt_kP


# verify the methods implemented above
if __name__ == '__main__':
    
    from .finiteMDPs import *

    np.random.seed(1)
    state_size = 50   
    discount = 0.99
    iter_no = 500
    #iter_no = 2000
    
    # MDP = FiniteMDP.FiniteMDP(state_size, ProblemType='randomwalk') # ,GarnetParam=(3,2))
    # MDP = FiniteMDP.FiniteMDP(state_size, ProblemType='garnet') # ,GarnetParam=(3,2))   
    # MDP = FiniteMDP.FiniteMDP(2,ActionSize = 1, ProblemType='TwoThreeStatesProblem',TwoThreeStatesParam = 0.95)
    # MDP = FiniteMDP.FiniteMDP(3,ActionSize = 1, ProblemType='TwoThreeStatesProblem')
    # MDP = FiniteMDP.FiniteMDP(3,ActionSize = 1, ProblemType='TwoThreeStatesProblem', TwoThreeStatesParam = 1/3)
    # MDP = FiniteMDP.FiniteMDP(state_size, ActionSize = 4, ProblemType = 'randomwalk', GarnetParam = (3, 5)) 
    MDP = FrozenLakeMDP(map_name = '4x4', is_slippery = True, render_mode = 'human', policy = 1)

    # pi = [1] * MDP.n_states # Policy that always chooses the first action (50 elements of 0's)
    # pi = np.random.randint(0, 2, MDP.n_states)
    pi = None # This is the greedy policy ==> Control case (in control I want to evaluate the optimnal policy associated to V*)
    
    # Some useful information
    P_shape = np.array(MDP.P).shape # P has shape 4 x 50 x 50
    R_shape = np.array(MDP.R).shape # R has shape 50 x 1
    print(f"The policy is {pi}")
    print(f"The state-transition model P has shape {P_shape} and the reward model R has shape {R_shape}")
    print(f"The state-transition model P is {MDP.P}")
    print(f"The reward model R is {MDP.R}")
    
    # Call the very first function implemented in this file
    print ('Computing the value function using the original VI ...')
    Vopt_true, Qopt_true, V_trace_orig_VI = \
        value_iteration(MDP.R, MDP.P, discount, IterationsNo = iter_no, policy = pi)
        
    print("The algorithm has finished running.")
    print(f"The optimal value function is {Vopt_true}")

     
    # Correct accelerated version
    Vopt_new, Qopt_new, V_trace_new, \
        Q_trace_new, dV_trace_new, dQ_trace_new, z_trace_new, \
        BE_Q_trace_new, BE_Q_integ_trace_new, gain_trace = \
        value_iteration_with_acceleration_new(MDP.R, MDP.P, discount, IterationsNo = iter_no,
                                                alpha = 0., accelation_type='PID',
                                                gain = Gain(1.,0,0.0,0.05,0.95),
                                                policy = pi,
                                                gain_adaptation=True,
                                                meta_lr = 0.05, normalization_eps = 1e-16,
                                                normalization_flag = 'BE2')
        

    
    error = np.linalg.norm(V_trace_orig_VI - Vopt_true, axis = 0, ord = np.inf)
    error_new = np.linalg.norm(V_trace_new - Vopt_true, axis = 0, ord = np.inf)  
    # error_conv_VI= np.linalg.norm(V_trace_conv_VI - Vopt_true, axis = 0, ord=error_norm_ord)
    figure(figsize = [6, 6])
    semilogy(error)
    semilogy(error_new,'--')
    xlabel('Iteration', fontsize = 15)
    ylabel('$||V_k - V*||_{\infty}$', fontsize = 15) # 'v_k' is actually 'V_trace'
    legend(['VI (original)','VI with acceleration'])
    
    
    k1, k2 = 10, min(1000, iter_no-1)
    eff_discount = np. exp( np.log(error[k2]/error[k1]) / (k2 - k1) )
    eff_discount_new = np. exp( np.log(error_new[k2]/error_new[k1]) / (k2 - k1) )    
    
    print ('Original discount & Effective discount factor & the new one:', \
                discount, eff_discount, eff_discount_new)
    
    print ('Original planning horizon & Effective planning horizon & the new one:', \
                1./(1. - discount), 1./(1. - eff_discount), 1./(1. - eff_discount_new))
    
    grid()
    if pi is None:
        title('Error behaviour in the Control case', fontsize = 20)
    else:
        title('Error behaviour in the Prediction case', fontsize = 20)
    show()
    

    
    
