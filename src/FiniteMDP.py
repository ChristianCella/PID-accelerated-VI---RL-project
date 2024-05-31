#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 16:47:49 2019

@author: Amir-massoud Farahmand
"""

import numpy as np

# TODO:
# 1) I should verify P and R.
# 2) Actions are only state-dependent. Add action-dependency



class FiniteMDP:

    # Initialize
    def __init__(self, StateSize = 5, ActionSize = 2, ProblemType = None,
                 GarnetParam = (1,1), TwoThreeStatesParam = None):
        
        self.StatesNo = StateSize
        self.ActionsNo = ActionSize

        # Initialize the transition probability matrix
#        print 'Initializing a finite Markov class of type ', ProblemType,' with StateNo = ',StateSize, 'and ActionsSize = ', ActionSize
        self.P = [np.matrix(np.zeros( (StateSize,StateSize))) for act in range(ActionSize)]
        
        self.R = np.zeros( (StateSize,1) )
        
                
        if ProblemType is 'garnet':
            b_P = GarnetParam[0] # branching factor
            b_R = GarnetParam[1] # number of non-zero rewards
#            Garnet = True
            
         
        if ProblemType == 'TwoThreeStatesProblem':
            
            if TwoThreeStatesParam is None:
                TwoThreeStatesParam = 0.0

            self._TwoThreeStateProblem(p = TwoThreeStatesParam)
            return
        
        # Setting up the transition probability matrix
        for act in range(ActionSize):
            for ind in range(StateSize):
                pVec = np.zeros(StateSize)
                
                
                if ProblemType is 'garnet':
                    # Garnet-like (not exact implementaiton).
                    p_vec = np.append(np.random.uniform(0,1,b_P - 1),[0,1])
                    p_vec = np.diff(np.sort(p_vec))
                    pVec[np.random.choice(StateSize,b_P, replace = False)] = p_vec#[:,np.newaxis]

                    # pVec[np.random.choice(StateSize,b_P, replace = False)] = 0.1 * np.random.exponential(0.1,b_P)
                    
                    # if act == 0:
                    #     pVec[ (ind + 1) % StateSize ] = 0.2 # Walking to the right!
                    # else:
                    #     pVec[ (ind - 1) % StateSize ] = 0.2 # Walking to the left!
                elif ProblemType is 'randomwalk':
                    if act == 0:
                        pVec[ (ind + 1) % StateSize ] = 0.7 # Walking to the right!
                        pVec[ ind ] = 0.2
                        pVec[ (ind - 1) % StateSize ] = 0.1
                    else:
                        pVec[ (ind - 1) % StateSize ] = 0.7 # Walking to the left!
                        pVec[ ind ] = 0.2                        
                        pVec[ (ind + 1) % StateSize ] = 0.1
                # XXX I am not sure if it works properly!
                elif ProblemType is 'smoothrandomwalk':
                    if act == 0:
                        pVec[ min(ind + 1,StateSize):min(ind+5,StateSize) ] = 0.7/5 # Walking to the right!
                        pVec[ ind ] = 0.2/5
                        pVec[ max(ind - 5,0): ind] = 0.1/5
                        
#                        pVec[ (ind + 1) % StateSize:(ind+5) % StateSize ] = 0.7/5 # Walking to the right!
#                        pVec[ ind ] = 0.2/5
#                        pVec[ (ind - 5) % StateSize: (ind) % StateSize ] = 0.1/5
                    else:
                        pVec[ max(ind - 5,0): ind] = 0.7/5 # Walking to the left!
                        pVec[ ind ] = 0.2/5
                        pVec[ min(ind + 1,StateSize):min(ind+5,StateSize) ] = 0.1/5                        
                        
#                        pVec[ (ind - 5) % StateSize: (ind) % StateSize ] = 0.7/5 # Walking to the left!
#                        pVec[ ind ] = 0.2/5
#                        pVec[ (ind + 1) % StateSize:(ind+5) % StateSize ] = 0.1/5                        
                elif ProblemType is None:
                    pVec = np.random.exponential(1,StateSize)#*range(StateSize)
                
                
##                if Garnet == -1:
##                    if act == 0:
##                        pVec[ (ind + 1) % StateSize ] = 0.7 # Walking to the right!
##                        pVec[ ind ] = 0.2
##                        pVec[ (ind - 1) % StateSize ] = 0.1
##                    else:
##                        pVec[ (ind - 1) % StateSize ] = 0.7 # Walking to the left!
##                        pVec[ ind ] = 0.2                        
##                        pVec[ (ind + 1) % StateSize ] = 0.1                        
#                
#                if Garnet:
#                    # Garnet-like (not exact implementaiton). It walks to the right.
#                    pVec[random.choice(StateSize,b_P, replace = False)] = 0.1 * random.exponential(0.1,b_P)
#                    if act == 0:
#                        pVec[ (ind + 1) % StateSize ] = 0.2 # Walking to the right!
#                    else:
#                        pVec[ (ind - 1) % StateSize ] = 0.2 # Walking to the right!                        
#                else:
#                    # ORIGINAL ONE
#                    pVec = random.exponential(1,StateSize)#*range(StateSize)
#                    #            pVec = random.lognormal(0,1.5,StateSize)                
#                    #            # XXX Remember to remove. This is here to skew the distribution
#                    #            pVec = sort(pVec)
                
#                print(sum(pVec))
                pVec /= sum(pVec)
                self.P[act][ind,:] = pVec
            
        # Setting up the reward function
        if ProblemType is 'garnet':
            # self.R[np.random.choice(StateSize,b_R, replace = False)] = np.random.exponential(0.1,(b_R,1) )
            self.R[np.random.choice(StateSize,b_R, replace = False)] = np.random.uniform(0,1,b_R)[:,np.newaxis]
            # self.R = np.random.uniform(0,1,StateSize)[:,np.newaxis]
        elif ProblemType is 'randomwalk':
#            self.R[0:3] = 1. #1.
#            self.R[-3:] = 1. #1.
            # self.R[int(StateSize*0.45):int(StateSize*0.55)] = 1 # -1.
#            self.R[45:55] = 1.1
            # self.R[0:StateSize] = 1. # XXX NEW! 
            # self.R[0:StateSize:5] = 1.
            # self.R[1:StateSize:2] = -1
            self.R[10] = -1. # XXX Used to be -1 XXX
            self.R[-10] = 1. # XXX Temporarily commented! XXX
        elif ProblemType is 'smoothrandomwalk':
            self.R[0:3] = 1.
            self.R[-3:] = 1.
            self.R[int(StateSize*0.45):int(StateSize*0.55)] = 1.1            
#            self.R[45:55] = 1.1
        else:
            self.R = np.random.uniform(0,1,StateSize)     
            
#        self.R = np.random.uniform(0,1,StateSize) # XXX Make sure this is not ON all the time!         

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


            # P = [[0.0, 0.0, 1.],
            #       [1., 0.0, 0.],
            #       [0.1, 0.9, 0.0]]

            # P = [[0, 1, 0],
            #       [0, 0, 1],
            #       [0.9, 0.0, 0.1]]

            # This is interesting enough. The complex eigenvalues
            # are on the unit circle.
            # P = [[0, 1, 0],
            #       [0, 0, 1],
            #       [1, 0, 0]]

            # P = [[0.1, 0.9, 0],
            #       [0, 0.1, 0.9],
            #       [0.9, 0, 0.1]]



        P = np.matrix(P)
        P = P/np.sum(P,axis = 1)

        for act in range(self.ActionsNo):
            self.P[act] = P

        # self.R[0:self.StatesNo-1] = -1
        self.R[0] = 1
        # self.R[2] = -1
        self.R[self.StatesNo-1] = -0.98 # -0.98 # -1 ORIGINAL

  
        
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
            
            
            
    

        
if __name__ == '__main__':

    from matplotlib.pyplot import *
    
    # Markov = FiniteMDP(25,ProblemType='randomwalk') #, Garnet=(2,2))
    
    Markov = FiniteMDP(25,ProblemType='garnet', GarnetParam=(5,2) )

    # Markov = FiniteMDP(3,ActionSize = 1, ProblemType='TwoThreeStatesProblem',
    #                     TwoThreeStatesParam = 0.4)
#    print 'P:', Markov.P
#    print 'R:', Markov.R
    
    imshow(Markov.P[0],interpolation = 'None')
    colorbar()
    figure()
    plot(Markov.R)

    #print Markov.DrawSamplesFromState(SamplesNo=3,x0=0)

#    X, XNext = Markov.DrawSamples(3,NextStateSamplesNo=5)
#    print 'X', X
#    print 'XNext', XNext

