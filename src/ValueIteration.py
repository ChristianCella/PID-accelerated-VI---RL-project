#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 16:46:48 2019

@author: Amir-massoud Farahmand
"""

# Note: The accelerated VI function is value_iteration_with_acceleration_new and
# the gain adaptation mechanism is adapt_gain_new.
# The version without _new as the suffix is a bit older, and should be removed at some point.


import numpy as np
from matplotlib.pyplot import *

from collections import namedtuple

# I_beta determines the root of the averager (integrator);
# I_alpha determines its gain.
# Gain = namedtuple('Gain', ['kP', 'kD', 'kI', 'I_alpha', 'I_beta'], defaults=[1,0,0,0,0])
Gain = namedtuple('Gain', ['kP', 'kI', 'kD', 'I_alpha', 'I_beta'], defaults=[1,0,0,0,0])


#def value_iteration(R,P,discount,IterationsNo = None):
#    
#    state_size = shape(P)[0]
#    
#    R = R.reshape(state_size,1)
#    
#    V = zeros( (state_size,1) )
#        
#    if IterationsNo == None:
#        IterationsNo = int(10/(1-discount))
#        
##    print IterationsNo
#
#    # Debugging
#    VTrace = zeros( (state_size,IterationsNo))
#    
##    print shape(VTrace)
#    
#    for iter in range(IterationsNo):
##        P*V
##        print shape(P*V), shape(R)
#        V = R + discount*P*V        
#        VTrace[:,iter] = V.reshape( (state_size,) )
#    
#    return V, VTrace


# Value Iteration for finite state and action MDP
def value_iteration(R,P,discount,IterationsNo = None, policy = None):
    """Value Iteration for finite state and action spaces.
    """    
        
    action_size, state_size = np.shape(P)[0:2]
    
    R = R.reshape(state_size,1)        
    V = np.zeros( (state_size,1) )
    Q = np.matrix(np. zeros( (state_size,action_size) ) )
        
    if IterationsNo == None:
        IterationsNo = int(10/(1-discount))
        
    # Debugging
    VTrace = np.zeros( (state_size,IterationsNo))

    V = np.max(Q,axis = 1).reshape(state_size,1)
    
    for iter in range(IterationsNo):
        
        for act in range(action_size):
            Q[:,act] = R + discount*P[act]*V
    
        if policy is None:
            V = np.max(Q,axis = 1).reshape(state_size,1)
        else:
            V = np.array([Q[m,policy[m] ] for m in range(state_size)]).reshape(state_size,1)
        
        VTrace[:,iter] = V.reshape( (state_size,) )
    
    return V, Q, VTrace    





# Value Iteration for finite state and action MDP
# This is an older version. Should be removed after verifying that I have included everything that 
# I wanted to be implement in the new version too.
def value_iteration_with_acceleration(R,P,discount,IterationsNo = None, policy = None, alpha = 0.0):
    """Value Iteration for finite state and action spaces.
    """    
        
    action_size, state_size = np.shape(P)[0:2]
    
    R = R.reshape(state_size,1)        
    V = np.zeros( (state_size,1) )
    Q = np.matrix(np. zeros( (state_size,action_size) ) )
        
    if IterationsNo == None:
        IterationsNo = int(10/(1-discount))
        
    # Debugging
    V_trace = np.zeros( (state_size,IterationsNo))

    V = np.max(Q,axis = 1).reshape(state_size,1)
    
    
#        V_tmp = V
#        V = r + gamma*P*V + alpha*dV
##        V = r + gamma*P*(V + alpha*dV)
#        dV = V - V_tmp
#        V_trace.append(dV)

#    alpha = 0.3 * discount    
    alpha *= discount
    
    dQ = 0*Q
    dQ_trace = []
    
    dV = 0*V
    dV_trace = []
    
    
    for iter in range(IterationsNo):
        
        Q_tmp = np.copy(Q)
        V_tmp = np.copy(V)
        for act in range(action_size):
           Q[:,act] = R + discount*P[act]*V + alpha*dQ[:,act] # Original version
            # Q[:,act] = R + discount*P[act]*V + alpha*dV # Experimental version
      
        dQ = Q - Q_tmp
        # XXX
#        dQ = np.clip(dQ, -1./(1+log(iter)), 1./(1+log(iter)))
#        dQ = np.clip(dQ, -5., +5.)

        # Idea: Is this clipping somehow similar to the dynamics of optimization of a DNN?
        
#        print dQ
        dQ_trace.append(dQ)
    
        
    
        if policy is None:
            V = np.max(Q,axis = 1).reshape(state_size,1)
        else:
            V = np.array([Q[m,policy[m] ] for m in range(state_size)]).reshape(state_size,1)

        # XXX
        dV = V - V_tmp
        
        # XXX
        # dV = np.clip(dV, -1./(1+np.log(1+iter)), 1./(1+np.log(1+iter)))        
        # dV = np.clip(dV, -1., 1.0)
        dV_trace.append(dV)

        
        V_trace[:,iter] = V.reshape( (state_size,) )

    dQ_trace = np.array(dQ_trace)
    dV_trace = np.array(dV_trace).squeeze()    
    
    
    return V, Q, V_trace, dQ_trace, dV_trace




# Accelerated Value Iteration for finite state and action MDP
# This is a research code, and has a lot of code for debugging and tracking.
# Excuse the appearance!
def value_iteration_with_acceleration_new(R,P,discount,IterationsNo = None, policy = None, alpha = 0.0,
                                            accelation_type = None, gain = Gain(1.,0,0),
                                            gain_adaptation = False, meta_lr = None,
                                            normalization_flag = 'BE2', normalization_eps = 1e-6):
    """Value Iteration for finite state and action spaces.
    It supports accelerated variants, such as the PID one.
    This is used for the paper.
    """    
    

    action_size, state_size = np.shape(P)[0:2]
    
    R = R.reshape(state_size,1)        
    V = np.zeros( (state_size,1) )
    Q = np.matrix(np. zeros( (state_size,action_size) ) )
        
    # Simple heuristic to chooce the iteration number, if not specified.
    if IterationsNo == None:
        IterationsNo = int(10/(1-discount))
    
    # print(gain) # XXX Remove! XXX

    # Construct an approximate Phat
    Phat = []
    for a in range(action_size):
        # p_hat = np.matrix( np.diag(np.diag(P[a])) )
        p_hat = P[a]*(np.eye(state_size) + 0.2*np.random.rand(state_size, state_size) )
        p_hat = p_hat/np.sum(p_hat,axis = 1)
        Phat.append( p_hat)

    # print(Phat[0].shape)
    # imshow(Phat[0])
    # figure()
    # imshow(diag(diag(MDP.P[1])))

    # Gain adaptation
    kD_mom = 0
    kI_mom = 0
    kP_mom = 0

    # Debugging
    V_trace = np.zeros( (state_size,IterationsNo))
    Q_trace = []

    pi_trace = []

    V = np.max(Q,axis = 1).reshape(state_size,1)
    
    # alpha *= discount
    
    dQ = 0*Q
    dQ_trace = []
    
    dV = 0*V
    dV_trace = []
    
    z_trace = []
    
    

    # Controller
    K_d_mat = np.zeros_like(P) # Matrix gain (scalar will be multiplied)
    # print(gain.kD)
    
    for act in range(action_size):
        K_d_mat[act] = (0.0*P[act] + 1.0*np.eye(state_size))
        # K_d_mat[act] = (1.0*P[act] + 0.1*np.eye(state_size))


    lam = 1. # XXX

    BE_Q = 0*Q
    BE_V = 0*V
    beta = 0.9 # Parameter for BE averaging

    BE_Q_trace = []
    # BE_Q_imm_trace = []

    # These are the states of the integrator of PID
    BE_Q_integ = 0*BE_Q
    BE_V_integ = 0*BE_V

    BE_Q_integ_trace = []

    kD_trace = []
    kI_trace = []
    kP_trace = []

    kP_grad_trace = []
    kI_grad_trace = []
    kD_grad_trace = []
    gain_trace = []


    for iter in range(IterationsNo):
        
        # alpha *= 0.95

        # Experimental
        # if iter%10 == 0:
        #     policy = policy_greedy(Q)
        #     BE_Q_integ *= 0
        #     plot(V)


        if np.mod(iter,50) == -1:
            print(iter, alpha)
            
            # if iter > 50:
            #     alpha = 0

        Q_tmp = np.copy(Q)
        V_tmp = np.copy(V)

        # V += alpha*dV

        if accelation_type in {'P', 'PD', 'PI', 'PID'}:
            V_new, Q_new = Bellman_operator(R, P, discount, V, policy = policy)            
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

            # Q = (1-lam)*Q + lam*Q_new


        Q = (1-lam)*Q + lam*Q_new
        ## V = (1-lam)*V + lam*V_new

        # ToDo: This should be parametrized
        # BE_Q = 0.9*BE_Q + 0.1*(Q_tmp - Q_new)
        # BE_Q = 0.9*BE_Q + 0.1*(Q_tmp - Q_new) # ORIGINAL

        # BE_Q = beta*BE_Q + (1-beta)*(Q_new- Q_tmp)

        # BE_Q_tmp = (Q_new- Q_tmp)
        # # print(np.sign(np.sum(BE_Q)))
        # if np.sum(BE_Q) *  np.sum(BE_Q_tmp) > 1e-3:
        #     print('Hey!', np.sum(BE_Q), np.sum(BE_Q_tmp) )
        #     BE_Q *= 0

        # This is the one whose state space model is written in root_locus_controllers
        # BE_Q = beta*BE_Q + (1. - beta)*(Q_new- Q_tmp)

        BE_Q = Q_new - Q_tmp # Bellman error
         # Integration of BE. This is the same as denoted by z in my notes.
        BE_Q_integ = gain.I_beta * BE_Q_integ + gain.I_alpha * BE_Q

        # This one has a different constant
        # BE_Q = beta*BE_Q + 1.*(Q_new- Q_tmp)
        # BE_Q = BE_Q + (Q_new- Q_tmp)
        
        # Tracing
        # BE_Q_imm_trace.append( Q_new- Q_tmp )
        # BE_Q_trace.append(BE_Q)

        # Tracing of the Bellman Error and its temporally averaged one
        BE_Q_trace.append(BE_Q)
        BE_Q_integ_trace.append(BE_Q_integ)


        # BE_Q += (Q_new - Q_tmp)

        # BE_V = 0.9*BE_V + 0.1*(V_tmp - V_new)
        # BE_V = beta*BE_V + (1-beta)*(V_new - V_tmp)     # ORIGINAL

        BE_V = V_new - V_tmp
        BE_V_integ = gain.I_beta*BE_V_integ + gain.I_alpha*BE_V


        # Experimental: Resetting BE_Q_integ and BE_V_integ
        # if iter%50 == 0:
        #     BE_V_integ *= 0
        #     BE_Q_integ *= 0

    

        # V, Q = Bellman_operator(R, P, discount, V + alpha*dV, policy = policy)

        # V, Q = Bellman_operator(R, P, discount, V) #, policy = policy)

        # Q += alpha*dV
        # Q = Q + alpha*dQ
        # Q += alpha*dQ

        dQ_corr = 0*dQ
        for act in range(action_size):
        #    dQ_corr[:,act] = P[act]*dQ[:,act]
            # dQ_corr[:,act] = dQ[:,act]        
            dQ_corr[:,act] = K_d_mat[act]*dQ[:,act]
            # print(np.shape(dQ[:,act]), np.shape(Q_new[:,act] - Q_tmp[:,act]))
            # print( np.linalg.norm( dQ[:,act] - np.matrix((Q_tmp[:,act] - Q_trace[-1][:,act])) ) )
            # if iter > 2:
            #     # print( np.linalg.norm( dQ[:,act] - (Q_tmp[:,act]  - Q_trace[-2][:,act]) ) )
            #     # dQ_corr[:,act] = K_d_mat[act]*np.matrix(Q_trace[-1][act] - Q_trace[-1][act])

            #     # print(np.shape(Q_trace[-1][:,act,newaxis]))
            #     # print(np.shape(dQ[:,act]))
            #     print( np.linalg.norm( dQ[:,act] - (Q_tmp[:,act,newaxis]  - Q_trace[-2][:,act,newaxis]) ) )
            #     # print( np.shape(dQ[:,act]))
            #     # print( np.linalg.norm(Q_tmp[:,act,newaxis]  - Q_trace[-1][:,act,newaxis]) )
            # dQ_corr[:,act] = K_d_mat[act]*(Q_new[:,act] - Q_tmp[:,act])
            # dQ_corr[:,act] = np.matmul(K_d_mat[act],dQ[:,act])


        # print(np.linalg.norm(Q - Q_new))
        
        if accelation_type == 'P':
            Q = (1 - gain.kP)*Q_tmp + gain.kP*Q_new

        if accelation_type == 'PD':
            # Derivative controller term
            # Q += alpha*dQ_corr # Original
            Q = (1 - gain.kP)*Q_tmp + gain.kP*Q_new + gain.kD*dQ_corr 
            # print('PD controller')

        if accelation_type == 'PID':
            # D + I part of the controller (The Bellman opertor is the P part)
            # Q += 1.*alpha*BE_Q  + 0*(+1.)*alpha*dQ_corr
            # Q += alpha*BE_Q  # Integrative term
            # Q += 0.0*dQ_corr # Derivative term
            # Q += 0*(+1.)*alpha*dQ_corr
            # print('PID controller')
            Q = (1 - gain.kP)*Q_tmp + gain.kP*Q_new + gain.kD*dQ_corr + gain.kI*BE_Q_integ 

        # print np.linalg.norm(dQ - dQ_corr)

        # print np.shape(dQ)

        # for act in range(action_size):
        #    Q[:,act] += alpha*dQ[:,act]


        dQ = Q - Q_tmp

    
        # Tracing
        
    
        if policy is None:
            V = np.max(Q,axis = 1).reshape(state_size,1)
        else:
            V = np.array([Q[m,policy[m] ] for m in range(state_size)]).reshape(state_size,1)

        # XXX
        dV = V - V_tmp

        # Gain adaptation
        if gain_adaptation and iter > 1: # and iter%5 == 0:
            V_km1 = V_trace[:,iter-1].reshape( (state_size,1) )
            V_km2 = V_trace[:,iter-2].reshape( (state_size,1) )

            Q_km1 = Q_trace[iter-1]
            Q_km2 = Q_trace[iter-2]
            
            # print('Q_km1:', np.shape(Q_km1))

    
            # print('||V_k - V_{k-1}||=', np.linalg.norm(V_k - V_km1))
            # print('||V_{k+1} - V_{k}||=', np.linalg.norm(V - V_k))
            # adapt_gain()
            # kD_grad = adapt_gain(R, P, discount, policy, V, V_km1, V_km2, z_k = BE_Q_integ)
            
            # Original and working for PE
            # kD_grad, kI_grad, kP_grad = adapt_gain(R, P, discount, policy, V, V_km1, V_km2, z_k = BE_Q_integ)

            # kD_grad_orig = kD_grad 
            # kI_grad_orig = kI_grad

            # Experimental XXX
            # normalization_eps = 0.97**iter

            # policy = [0]*state_size # EXPERIMENTAL --- BE CAREFUL.
            kD_grad, kI_grad, kP_grad = adapt_gain_new(R, P, discount, policy,
                                                    Q_k = Q, V_k = V,
                                                    Q_km1 = Q_km1, V_km1 = V_km1,
                                                    Q_km2 = Q_km2, V_km2 = V_km2,
                                                    z_k = BE_Q_integ,
                                                    normalization_flag = normalization_flag,
                                                    normalization_eps = normalization_eps)
            
            # print(np.linalg.norm(kD_grad - kD_grad_orig))
            # print(np.linalg.norm(kI_grad - kI_grad_orig))

            # Just one heuristic
            # if policy:
            #     meta_lr = 1 - discount
            # else:
            #     meta_lr = (1 - discount)/2

            # discount = 0.999
            # meta_lr = 0.001 # 0.005 (PE), 0.001 (control)
            
            # discount = 0.99
            # meta_lr = 0.01 # 0.01 (PE), 0.005 (control)

            # meta_lr = 0.2/np.sqrt(100 + iter) # + 0.01*np.random.rand()
            # if iter > 400:
            #     meta_lr = 0.005
            # else:
            #     meta_lr = 0.01

            # meta_lr = 0.1/np.sqrt(1 + iter) #XXX
            # meta_lr = 0.1 #/(1 + iter)
            # GD Version (ORIGINAL)
            kD_new = gain.kD - meta_lr*kD_grad # + meta_lr*0.1*np.random.randn()
            kI_new = gain.kI - meta_lr*kI_grad
            kP_new = gain.kP - meta_lr*kP_grad

            # Experimental XXX
            # normalization_eps = min(1.02*normalization_eps, 1)

            # normalization_eps *= 0.99

            # Multiplicative form (Experimental)
            # kD_new = gain.kD*(1 - meta_lr*kD_grad) - 0.1*meta_lr*kD_grad # + meta_lr*0.1*np.random.randn()
            # kI_new = gain.kI*(1 - meta_lr*kI_grad) - 0.1*meta_lr*kI_grad
            # kP_new = gain.kP*(1 - meta_lr*kP_grad) - 0.1*meta_lr*kP_grad


            # GD + Momentum Version
            # mom_pole = 0.9
            # kD_mom = mom_pole*kD_mom + (1 - mom_pole)*kD_grad
            # kI_mom = mom_pole*kI_mom + (1 - mom_pole)*kI_grad
            # kP_mom = mom_pole*kP_mom + (1 - mom_pole)*kP_grad

            # kD_new = gain.kD - meta_lr*kD_mom # + meta_lr*0.1*np.random.randn()
            # kI_new = gain.kI - meta_lr*kI_mom
            # kP_new = gain.kP - meta_lr*kP_mom



            # kP_new = gain.kP*(1 - 1e-3*kP_grad) # This is from Martinez 2017, apparently.

            # if iter%50 == 0:
            #     kP_new = 1.
            #     kP_new = gain.kP - 0.1*kP_grad
            # else:
            #     kP_new = gain.kP

            # if iter%200 == 0:
            #     # Q *= 0.5
            #     # V *= 0.5
            #     # kP_new = 1.
            #     print("At iteration", iter, "the meta-learning rate and normalization_eps rate are:",
            #             meta_lr, normalization_eps)

                # normalization_eps = 1e-8 # XXX REMOVE IT LATER XXX

                # kP_new = 1.
                # kI_new *= 0.5
                # kD_new *= 0.5



            # kD_new = gain.kD - 0.01*kD_grad + 0.00*np.random.randn()
            # kI_new = gain.kI - 0.01*kI_grad + 0.00*np.random.randn()
            # kP_new = gain.kP - 0.01*kP_grad + 0.00*np.random.randn()

            
            # gain = Gain(gain.kP, gain.kI, kD_new, gain.I_alpha, gain.I_beta)
            gain = Gain(kP_new, kI_new, kD_new, gain.I_alpha, gain.I_beta)
            if iter%50 == -1:
                print(iter, gain)

            kD_trace.append(kD_new)
            kI_trace.append(kI_new)
            kP_trace.append(kP_new)

            gain_trace.append(gain)

            kP_grad_trace.append(kP_grad)
            kI_grad_trace.append(kI_grad)
            kD_grad_trace.append(kD_grad)




        # Tracing (which is also used for gain adaptation)
        dV_trace.append(dV)
        dQ_trace.append(dQ)
        Q_trace.append(np.copy(Q))
        V_trace[:,iter] = V.reshape( (state_size,) )

        pi_trace.append(policy_greedy(Q))

    dQ_trace = np.array(dQ_trace)
    dV_trace = np.array(dV_trace).squeeze()
    Q_trace = np.array(Q_trace)

    pi_trace = np.array(pi_trace)
    
    z_trace = np.array(z_trace).squeeze()
    BE_Q_trace = np.array(BE_Q_trace)
    BE_Q_integ_trace = np.array(BE_Q_integ_trace)
    # BE_Q_imm_trace = np.array(BE_Q_imm_trace)
    
    if gain_adaptation and False:
        figure()
        subplot(2,1,1)
        plot(kP_trace,'b', linewidth = 2)
        plot(kI_trace,'m', linewidth = 2)        
        plot(kD_trace,'r', linewidth = 2)
        legend(['k_p', 'k_I', 'k_d'])
        xlabel('Iteration', fontsize=20)
        ylabel('Controller gains', fontsize = 20)


        subplot(2,1,2)
        plot(np.log(np.abs(kP_grad_trace)),'b')
        plot(np.log(np.abs(kI_grad_trace)),'r')
        plot(np.log(np.abs(kD_grad_trace)),'m')

        # plot(np.log(np.abs(kP_grad_trace))*np.sign(kP_grad_trace),'b')
        # plot(np.log(np.abs(kI_grad_trace))*np.sign(kI_grad_trace),'r')
        # plot(np.log(np.abs(kD_grad_trace))*np.sign(kD_grad_trace),'m')

        # plot(kP_grad_trace,'b')
        # plot(kI_grad_trace,'m')
        # plot(kD_grad_trace,'r')
        legend(['k_p', 'k_I', 'k_d'])
        xlabel('Iteration', fontsize=20)
        ylabel('Log of gain derivatives', fontsize = 20)


        # Visualizing the the greedy policy throughout iterations
        # figure()
        # matshow(pi_trace.transpose())
        # xlabel('Iteration')
        # print(gain_trace)

    
    return V, Q, V_trace, Q_trace, dV_trace, dQ_trace, z_trace, BE_Q_trace, BE_Q_integ_trace, gain_trace



def Bellman_operator(R,P,discount, V = None, Q = None, policy = None):
    """Bellman operator for finite state and action spaces.
    """    
        
    action_size, state_size = np.shape(P)[0:2]
    
    R = R.reshape(state_size,1)        
    
    # V = np.zeros( (state_size,1) )
    Q_new = np.matrix(np.zeros( (state_size,action_size) ) )
    
    # If V isn't given, compute it using Q.
    if V is None:
        if policy is None:
            V = np.max(Q,axis = 1).reshape(state_size,1)
        else:
            V = np.array([Q[m,policy[m] ] for m in range(state_size)]).reshape(state_size,1)

    # print V
    # print('V:', np.shape(V))
    for act in range(action_size):
        Q_new[:,act] = R + discount*P[act]*V
    
    if policy is None:
        V_new = np.max(Q_new,axis = 1).reshape(state_size,1)
    else:
        V_new = np.array([Q_new[m,policy[m] ] for m in range(state_size)]).reshape(state_size,1)

    return V_new, Q_new


def policy_greedy(Q,x = None):
    """Return the greedy policy for an action-value function Q."""

    if x is None:
        x = range(np.shape(Q)[0])

    act = np.argmax(Q[x,:],axis = 1)
    
    return np.array(act).squeeze()


def adapt_gain_new(R, P, discount, policy, Q_k, V_k, Q_km1, V_km1, Q_km2, V_km2, z_k = None,
                   normalization_flag = True, normalization_eps = 1e-8,
                   truncation_flag = True, truncation_threshold = 1.):
    """ Gain adaptation for PID gains.
    It is based on gradient descent on the Bellman Error.
    This is used for the paper.
    """   

    if policy is None:
        policy_k = policy_greedy(Q_k)

    # print('||policy - policy_k||_1', np.linalg.norm(policy - policy_k, ord=1))

    # BE(V_k) is needed for all terms, so let's compute TV_k and TQ_k first
    TV_k, TQ_k = Bellman_operator(R, P, discount, V = V_k, policy = policy) # ORIGINAL

    TV_k = np.array(TV_k)
    TQ_k = np.array(TQ_k)
    BE_V_k = TV_k - V_k # Needed for the kD term
    BE_Q_k = TQ_k - Q_k

    if policy is None:
        BE_k = BE_Q_k
    else:
        BE_k = BE_V_k



    # BE(V_{k-1}) is needed for the kP term (and w.r.t. alpha; not implemented yet)
    # So we compute T V_{k-1}. This has already been computed in the pervious iteration
    # of VI, so re-computation isn't needed. But I do it again to simplify the book keeping.
    TV_km1, TQ_km1 = Bellman_operator(R, P, discount, V_km1, policy = policy)
    # TV_km1_tmp, TQ_km1_tmp = Bellman_operator(R, P, discount, Q = V_km1, policy = policy)


    TV_km1 = np.array(TV_km1)
    TQ_km1 = np.array(TQ_km1)
    BE_V_km1 = TV_km1 - V_km1
    BE_Q_km1 = TQ_km1 - Q_km1

    # print(np.linalg.norm(V_k1 - V_km1), np.linalg.norm(V_km1 - V_km2) )
    # print('V (k, k-1, k-2) shapes', V_k.shape, V_km1.shape, V_km2.shape)


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
        grad_J_wrt_kD =  np.sum(BE_V_k * grad_BE_wrt_kD) # ORIGINAL


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
        grad_J_wrt_kI =  np.sum(BE_k * grad_BE_wrt_kI) # ORIGINAL

    # Normalized version
    # grad_J_wrt_kI /= (np.linalg.norm(BE_k)*np.linalg.norm(grad_BE_wrt_kI) + 1e-3)
    # # grad_J_wrt_kI /= (np.linalg.norm(BE_k) + 1e-3) # Experimental

    # # Adding noise to see if it helps with the exploration
    # # grad_J_wrt_kI = np.clip(grad_J_wrt_kI, -0.2, +0.2) # + 0.001*np.random.randn()


    ################################################
    # Computing the gradient of BE(V_k) w.r.t. kP #
    ################################################

    # BE(V_{k-1}) is needed for the kP term (and w.r.t. alpha; not implemented yet)
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


    
    # Experimental
    # grad_J_wrt_kP /= (np.linalg.norm(BE_k, ord=1)*np.linalg.norm(grad_BE_km1_wrt_kP, ord=1) + 1e-1) # XXX

    # grad_J_wrt_kP =  np.sum(BE_k * grad_BE_km1_wrt_kP) # Corrected? 
    # grad_J_wrt_kP /= (np.linalg.norm(BE_k)*np.linalg.norm(grad_BE_km1_wrt_kP) + 1e-2)
    # grad_J_wrt_kP = np.clip(grad_J_wrt_kP, -0.1, +0.1) # + 0.*np.random.randn()




    # Normalization
    if normalization_flag:
        # eps = normalization_flag
        # NOTE: The following discussion is for original formulation, and not for BE2.
        # So these values should be ignored. I keep them here for now.
        # It is much better to have a larger normalizer for the gradient w.r.t. kP
        # 10^{-1} for kP and 10^-3 for kD and kI work well. Smaller values for
        # kD and kI (10^-6 or 10^-10) sometimes work much better, but make it closer
        # to be an unstable. It should be studied more.
        # The value of 1e-3 is used for the original NeurIPS 2020 submission.
        # But now I am changing it to a variable one.

        if normalization_flag == 'original':
        # ORIGINAL
        # This formulation is more finicky compared the BE2 one below. I keep it here for now.
            grad_J_wrt_kD /= (np.linalg.norm(BE_k)*np.linalg.norm(grad_BE_wrt_kD) + normalization_eps)
            grad_J_wrt_kI /= (np.linalg.norm(BE_k)*np.linalg.norm(grad_BE_wrt_kI) + normalization_eps)
            grad_J_wrt_kP /= (np.linalg.norm(BE_k)*np.linalg.norm(grad_BE_km1_wrt_kP) + normalization_eps)
        elif normalization_flag == 'BE2':
            # This actually works fine!
            # ORIGINAL formulation (for BE2, not the ORIGINAL above!). This is equivalent of taking the derivative of log(BE^2)
            # BE_squared = np.linalg.norm(BE_k)**2 + normalization_eps # XXX ORIGINAL XXX

            # This is what we present in the paper.
            # This is the derivative of the ratio of BE from k-1 to k.
            BE_squared = np.linalg.norm(BE_km1)**2 + normalization_eps # XXX With a newer normalization XXX
            # They don't seem to perform very differently, at least in not too small regime

            # print( np.linalg.norm(BE_k) - np.linalg.norm(BE_km1) )

            # BE_squared *= (1-discount)
            grad_J_wrt_kD /= BE_squared
            grad_J_wrt_kI /= BE_squared
            grad_J_wrt_kP /= BE_squared
        else:
            print('(adapt_gain_new) Incorrect choice of normalization!')


        # XXX Experimental XXX
        # grad_J_wrt_kD /= (np.linalg.norm(BE_k) + normalization_eps)
        # grad_J_wrt_kI /= (np.linalg.norm(BE_k) + normalization_eps)
        # grad_J_wrt_kP /= (np.linalg.norm(BE_k) + normalization_eps)




        # grad_J_wrt_kD /= (np.linalg.norm(BE_k)*np.linalg.norm(grad_BE_wrt_kD) + 1e-12)
        # grad_J_wrt_kI /= (np.linalg.norm(BE_k)*np.linalg.norm(grad_BE_wrt_kI) + 1e-12)
        # grad_J_wrt_kP /= (np.linalg.norm(BE_k)*np.linalg.norm(grad_BE_km1_wrt_kP) + 1e-12)



        # XXX Experimental XXX
        # grad_J_wrt_kD = np.sign(grad_J_wrt_kD)
        # grad_J_wrt_kI = np.sign(grad_J_wrt_kI)
        # grad_J_wrt_kP = np.sign(grad_J_wrt_kP)

        # XXX Experimental XXX
        # grad_J_wrt_kD = np.tanh(grad_J_wrt_kD)
        # grad_J_wrt_kI = np.tanh(grad_J_wrt_kI)
        # grad_J_wrt_kP = np.tanh(grad_J_wrt_kP)


        # XXX Experimental XXX
        # grad_J_wrt_kD = np.sign(grad_J_wrt_kD)*np.log(np.abs(grad_J_wrt_kD))
        # grad_J_wrt_kI = np.sign(grad_J_wrt_kI)*np.log(np.abs(grad_J_wrt_kI))
        # grad_J_wrt_kP = np.sign(grad_J_wrt_kP)*np.log(np.abs(grad_J_wrt_kP))


    # Truncation
    # truncation_flag = False
    if truncation_flag:
        # print('Hello!')
        grad_J_wrt_kD = np.clip(grad_J_wrt_kD, -truncation_threshold, truncation_threshold)
        grad_J_wrt_kI = np.clip(grad_J_wrt_kI, -truncation_threshold, truncation_threshold) # + 0.001*np.random.randn()
        grad_J_wrt_kP = np.clip(grad_J_wrt_kP, -truncation_threshold, truncation_threshold) # + 0.*np.random.randn()

        # grad_J_wrt_kD = np.clip(grad_J_wrt_kD, -0.2, +0.2)
        # grad_J_wrt_kI = np.clip(grad_J_wrt_kI, -0.2, +0.2) # + 0.001*np.random.randn()
        # grad_J_wrt_kP = np.clip(grad_J_wrt_kP, -0.2, +0.1) # + 0.*np.random.randn()


    # return np.sign(grad_J_wrt_kD)
    return grad_J_wrt_kD, grad_J_wrt_kI, grad_J_wrt_kP



# This is a working version, but it only implements the policy evaluation
# case. Use adapt_gain_new, which handles both policy evaluation and control.
def adapt_gain(R, P, discount, policy, V_k, V_km1, V_km2, z_k = None):

    # BE(V_k) is needed for all terms
    TV_k, TQ_k = Bellman_operator(R, P, discount, V_k, policy = policy)
    TV_k = np.array(TV_k) # XXX
    BE_k = TV_k - V_k # Needed for the kD term

    # BE(V_{k-1}) is needed for the kP term (and w.r.t. alpha; not implemented yet)
    TV_km1, TQ_km1 = Bellman_operator(R, P, discount, V_km1, policy = policy)
    TV_km1 = np.array(TV_km1) # XXX
    BE_km1 = TV_km1 - V_km1



    # print(np.linalg.norm(V_k1 - V_km1), np.linalg.norm(V_km1 - V_km2) )
    # print('V (k, k-1, k-2) shapes', V_k.shape, V_km1.shape, V_km2.shape)

    action_size, state_size = np.shape(P)[0:2]

    ################################################
    # Computing the gradient of BE(V_k) w.r.t. kD  #
    ################################################
    P_a_deltaV = np.matrix(np.zeros( (state_size,action_size) ) )
    
    deltaV = V_km1 - V_km2

    for act in range(action_size):
        P_a_deltaV[:,act] = P[act]*deltaV
    
    if policy is None:
        # This isn't based on any theory (as far as I know)
        # But I think I know how to do it right. Check the notes on 2020 May 9
        P_deltaV = np.max(P_a_deltaV,axis = 1).reshape(state_size,1)
    else:
        P_deltaV = np.array([P_a_deltaV[m,policy[m] ] for m in range(state_size)]).reshape(state_size,1)

    # ORIGINAL
    # PV = np.array([P_aQ[m,policy[m] ] for m in range(state_size)]).reshape(state_size,1)
    grad_BE_wrt_kD = discount*P_deltaV - deltaV



    # print('PV', PV.shape, 'grad_BE_wrt_kD', grad_BE_wrt_kD.shape, 'BE_kp1', BE_kp1.shape)

    # print('grad_BE_wrt_kD:', grad_BE_wrt_kD)
    # print('BE_k:', BE_k1)
    # print(PV)
    # print(BE_k1.shape, grad_BE_wrt_kD.shape)

    # np.array(BE_k1) * np.array(grad_BE_wrt_kD)
    # grad_J_wrt_kD =  np.sum(np.array(BE_k1) * np.array(grad_BE_wrt_kD))
    grad_J_wrt_kD =  np.sum(BE_k * grad_BE_wrt_kD) # ORIGINAL

    # Normalized version
    grad_J_wrt_kD/= (np.linalg.norm(BE_k)*np.linalg.norm(grad_BE_wrt_kD) + 1e-6)
    grad_J_wrt_kD = np.clip(grad_J_wrt_kD, -0.2, +0.2)
    # print('grad_J_wrt_kD:', grad_J_wrt_kD)
    # print(grad_J_wrt_kD.shape)

    ################################################
    # Computing the gradient of BE(V_k) w.r.t. kI  #
    ################################################
    P_a_z = np.matrix(np.zeros( (state_size,action_size) ) )
    
    # z_k is of state-action dimension. This is the state-dimension version of it.
    z_k_V = np.array([z_k[m,policy[m] ] for m in range(state_size)]).reshape(state_size,1)

    for act in range(action_size):
        P_a_z[:,act] = P[act]*z_k_V
    
    if policy is None:
        # This isn't based on any theory (as far as I know)
        # But I think I know how to do it right. Check the notes on 2020 May 9
        print("This hasn't been derived yet!")
        # P_z = np.max(P_a_z,axis = 1).reshape(state_size,1)
    else:
        P_z = np.array([P_a_z[m,policy[m] ] for m in range(state_size)]).reshape(state_size,1)

    grad_BE_wrt_kI = discount*P_z - z_k_V

    grad_J_wrt_kI =  np.sum(BE_k * grad_BE_wrt_kI) # ORIGINAL

    # Normalized version
    grad_J_wrt_kI /= (np.linalg.norm(BE_k)*np.linalg.norm(grad_BE_wrt_kI) + 1e-6)
    # Adding noise to see if it helps with the exploration
    grad_J_wrt_kI = np.clip(grad_J_wrt_kI, -0.2, +0.2) # + 0.*np.random.randn()

    # print('grad_J_wrt_kI:', grad_J_wrt_kI)
    # print(grad_J_wrt_kI.shape)


    ################################################
    # Computing the gradient of BE(V_k) w.r.t. kP #
    ################################################
    P_a_BE_km1 = np.matrix(np.zeros( (state_size,action_size) ) )
    
    for act in range(action_size):
        P_a_BE_km1[:,act] = P[act]*BE_km1
    
    if policy is None:
        # This isn't based on any theory (as far as I know)
        # But I think I know how to do it right. Check the notes on 2020 May 9
        print("This hasn't been derived yet!")
        # P_z = np.max(P_a_z,axis = 1).reshape(state_size,1)
    else:
        P_BE_km1 = np.array([P_a_BE_km1[m,policy[m] ] for m in range(state_size)]).reshape(state_size,1)

    grad_BE_km1_wrt_kP = discount*P_BE_km1 - BE_km1

    # I think this original formulation was incorrect. It should be BE_k instead of BE_km1.
    # grad_J_wrt_kP =  np.sum(BE_km1 * grad_BE_km1_wrt_kP) # ORIGINAL
    # Normalized version
    # grad_J_wrt_kP /= (np.linalg.norm(BE_km1)*np.linalg.norm(grad_BE_km1_wrt_kP) + 1e-1)

    # I suspect this is the correct one.
    grad_J_wrt_kP =  np.sum(BE_k * grad_BE_km1_wrt_kP) # Corrected? 
    grad_J_wrt_kP /= (np.linalg.norm(BE_k)*np.linalg.norm(grad_BE_km1_wrt_kP) + 1e-6)

        
    # # Adding noise to see if it helps with the exploration
    # grad_J_wrt_kP = np.clip(grad_J_wrt_kP, -0.2, +0.2) # + 0.*np.random.randn()

    # print('grad_J_wrt_kP:', grad_J_wrt_kP)
    # print(grad_J_wrt_kP.shape)

    # return np.sign(grad_J_wrt_kD)
    return grad_J_wrt_kD, grad_J_wrt_kI, grad_J_wrt_kP







if __name__ == '__main__':
    
    import FiniteMDP
    
    # state_size = 50
    # MDP = FiniteMDP.FiniteMDP(state_size, ProblemType='randomwalk') # ,GarnetParam=(3,2))
# #    MDP = FiniteMDP.FiniteMDP(state_size, ProblemType='garnet') # ,GarnetParam=(3,2))
    
    # MDP = FiniteMDP.FiniteMDP(2,ActionSize = 1, ProblemType='TwoThreeStatesProblem',TwoThreeStatesParam = 0.95)
    # MDP = FiniteMDP.FiniteMDP(3,ActionSize = 1, ProblemType='TwoThreeStatesProblem')
    # MDP = FiniteMDP.FiniteMDP(3,ActionSize = 1, ProblemType='TwoThreeStatesProblem', TwoThreeStatesParam = 1/3)

    np.random.seed(1)
    state_size = 30
    MDP = FiniteMDP.FiniteMDP(state_size, ActionSize = 4, ProblemType='garnet', GarnetParam=(3,5)) 

    # state_size = 20 #P.shape[0]
    # P = np.eye(state_size)
    # # P = np.random.random( (state_size,state_size) )    
    # P = np.matrix(P)
    # P = P/np.sum(P,axis = 1)
   
    # R = np.zeros(state_size)
    # # R[int(state_size/2)] = 1.
    # # R[0] = -1.
    # R = np.linspace(0,1,state_size)

    # MDP = FiniteMDP.FiniteMDP(state_size, 2)
    # MDP.P[0] = P
    # MDP.P[1] = P    
    # MDP.R = R

    # MDP.R += np.random.rand(state_size,1)

    discount = 0.99

    iter_no = 1000
    
    pi = [0]*state_size
    # pi = np.random.randint(0,2,state_size)
    # pi = None
    
    
    
    
    print ('Computing the value function using the original VI ...')
    Vopt_true, Qopt_true, V_trace_orig_VI = \
        value_iteration(MDP.R, MDP.P, discount, IterationsNo=10*iter_no, policy=pi)
        
    print ('Computing the value function using the accelerated VI ...')
    Vopt, Qopt, V_trace, dQ_trace, dV_trace = \
        value_iteration_with_acceleration(MDP.R, MDP.P, discount, IterationsNo=iter_no,
                                          alpha = 0.0, policy=pi) 
    # V, Q, V_trace, Q_trace, dV_trace, dQ_trace, z_trace, BE_Q_trace, BE_Q_integ_trace, gain_trace
    Vopt_new, Qopt_new, V_trace_new, \
        Q_trace_new, dV_trace_new, dQ_trace_new, z_trace_new, \
        BE_Q_trace_new, BE_Q_integ_trace_new, gain_trace = \
        value_iteration_with_acceleration_new(MDP.R, MDP.P, discount, IterationsNo=iter_no,
                                                alpha = 0., accelation_type='PID',
                                                gain = Gain(1.,0,0.0,0.05,0.95),
                                                policy = pi,
                                                gain_adaptation=True,
                                                meta_lr = 0.05, normalization_eps = 1e-16,
                                                normalization_flag = 'BE2')
        
    # import sys
    # sys.exit(0)

    figure()
    subplot(1,2,1)
    semilogy(np.linalg.norm(BE_Q_integ_trace_new[:,:,0], axis = 1),'b')
    plot(np.linalg.norm(BE_Q_trace_new[:,:,0], axis = 1),'k')
    xlabel('Iterations')
    ylabel('|| BE_Q ||')
    legend(['Integrated BE', 'Immediate BE'])

    subplot(1,2,2)    
    state_no = 5
    plot(BE_Q_integ_trace_new[:,state_no,0],'b');
    plot(BE_Q_trace_new[:,state_no,0],'k');    
    figure()
    # xlabel('Iterations')
    # ylabel('BE_Q (at state )'+str(state_no))
    # legend(['Integrated BE', 'Immediate BE'])

    # Visualizing the optimal Q
    # figure()
    # # plot(Qopt,lw=2)
    # plot(Qopt[:,0],lw=2, label='$Q(\cdot,R)$')
    # plot(Qopt[:,1],lw=2, label='$Q(\cdot,L)$')
    # xlabel('States', fontsize=20)
    # ylabel('Value', fontsize=20)
    # legend()
    


#    plot(Qopt,linewidth = 2.0)
#    plot(Vopt, 'k')
#    plot(policy_greedy(Qopt) , 'o')    
#    figure()
    
#    plot(V_trace)
#    dV = np.diff(V_Trace, axis = 1)
#    plot(dV[:,::10])
#    legend(['1','2','3'])
    
#    policy_greedy(Qopt)
    
    error = np.linalg.norm(V_trace - Vopt_true, axis = 0, ord=np.inf)
    error_new = np.linalg.norm(V_trace_new - Vopt_true, axis = 0, ord=np.inf)    
        
    
    # Important visualization
    # semilogy(np.linalg.norm(V_trace - Vopt_true, axis = 0, ord=np.inf))
    # semilogy(np.linalg.norm(V_trace_new - Vopt_true, axis = 0, ord=np.inf),'--') 
    figure()
    semilogy(error)
    semilogy(error_new,'--')

    xlabel('Iteration')
    ylabel('||V_trace - Vopt_true||')
    legend(['VI (original)','VI with acceleration'])
    
    
    k1, k2 = 10, min(1000, iter_no-1)
    eff_discount = np. exp( np.log(error[k2]/error[k1]) / (k2 - k1) )
    eff_discount_new = np. exp( np.log(error_new[k2]/error_new[k1]) / (k2 - k1) )    
    print ('Original discount & Effective discount factor & the new one:', \
                discount, eff_discount, eff_discount_new)
    print ('Original planning horizon & Effective planning horizon & the new one:', \
                1./(1. - discount), 1./(1. - eff_discount), 1./(1. - eff_discount_new))
    
#    figure()    
#    plot(np.linalg.norm(dV_trace,axis = 1, ord=np.inf))
#    plot(np.linalg.norm(dV_trace_new,axis = 1, ord=np.inf),'--')    
#    xlabel('Iteration')
#    ylabel('dV_trace')
    
    
#    Bellman_operator(MDP.R, MDP.P, discount, V = Vopt)
    
    # figure()
    # subplot(1,2,1)
    # plot(V_trace.transpose())    
    # subplot(1,2,2)
    # plot(V_trace_new.transpose())
    # xlabel('Iteration')
    # ylabel('V_trace')
    
#    figure()
#    plot(dV_trace_new[:50,0:5]);

#    state_ind = 2
##    plot(Q_trace_new[:,state_ind,0],'b');
#    plot(BE_Q_trace_new[:,state_ind,0],'k');
#    plot(BE_Q_imm_trace_new[:,state_ind,0],'g--')

    
#    plot((V_trace-V_trace_new).transpose())
    
#    figure()
#    semilogy(np.linalg.norm(z_trace_new, axis = 1))
#    xlabel('Iteration')
#    ylabel('|| linear_predict - TV ||')
#    
#    figure()
#    plot(z_trace_new[10,:])
    