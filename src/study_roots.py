#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 13:55:10 2019

@author: sologen
"""

from numpy import *
from numpy.linalg import eig
from numpy.linalg import inv

from scipy.optimize import basinhopping


from matplotlib.pyplot import *


def root_3rd(alpha):
#    return roots([1, -(gamma+alpha[0]), -alpha[0], (alpha[0]+alpha[1])] )
    return roots([1.0, -(gamma+alpha[0]), -alpha[1], (alpha[0]+alpha[1])] )    


def root_2nd(alpha, gam = gamma):
    return roots([1.0, -(gam+alpha), +alpha] )    


def A_mat_2nd(alpha):
    A = vstack([        
            hstack([gamma*P + alpha*eye(state_size), - alpha * eye(state_size)]),
            hstack([eye(state_size), zeros((state_size,state_size))])
            ]
        )
    
    return A
    
def A_mat_3rd(alpha_0, alpha_1):
    A = vstack([        
            hstack([gamma*P + alpha_0*eye(state_size), alpha_1 * eye(state_size), -(alpha_0 + alpha_1)*eye(state_size)]),
            hstack([zeros((state_size,state_size)), eye(state_size), zeros((state_size,state_size))]),
            hstack([zeros((state_size,state_size)), zeros((state_size,state_size)), eye(state_size)])        
            ]
        )
    
    return A

def root_2nd_robust_mag(alpha, gamma):
    
    d_range = exp(2j*pi*linspace(0,1))
    
    
#    root_as_fn_d = abs(array([root_2nd(alpha, d*gamma) for d in d_range]))
    root_as_fn_d = (array([root_2nd(alpha, d*gamma) for d in d_range]))    
    
#    return np.max(root_as_fn_d[:,0])
#    return np.max(root_as_fn_d)    
    return root_as_fn_d #, np.max(root_as_fn_d)
    

    
state_size = 2 #P.shape[0]
#P = [[0.95,0.05],[0.03, 0.97] ]
#P = [[1.0]]
P = random.random( (state_size,state_size) )
P = matrix(P)
#P = (P + P.transpose())/2
P = P/sum(P,axis = 1)





gamma = 0.99

#print root_3rd([0.2,-0.2])

root_3rd_largest_root = lambda x: max(abs(root_3rd(x)))
root_2nd_largest_root = lambda x: max(abs(root_2nd(x, gamma)))

##res = basinhopping(root_3rd_largest_root,[0,0])
#res = basinhopping(root_2nd_largest_root,[0])
#print 'Solution:', res.x
#print 'Largest root:', root_2nd_largest_root(res.x)

#alpha_0, alpha_1 =  res.x
##alpha_0, alpha_1 = -0.21, 0.18
#pol = [1.0, -(gamma+alpha_0), -alpha_1, (alpha_0+alpha_1)]
#
#alpha = res.x

#A = A_mat_2nd(alpha)
#print A
#D, _ = eig(A)
#print 'Eigenvalues:', D

D, V = eig(P)

alpha_range = linspace(0,1,100)
root_sol = array([root_2nd_largest_root(alpha) for alpha in alpha_range])
root_all_sol = array([root_2nd(alpha) for alpha in alpha_range])
#root_all_sol = array([root_2nd(alpha, D[0]*gamma) for alpha in alpha_range])

eig_sol_all = array([ eig(A_mat_2nd(alpha))[0] for alpha in alpha_range])
eig_sol = array([ max(abs(eig(A_mat_2nd(alpha))[0])) for alpha in alpha_range])

plot(alpha_range, root_sol,'b')
plot(alpha_range, eig_sol, 'r')
#figure()
#plot(alpha_range, eig_sol_all.imag, '.')
#plot(alpha_range, 0*eig_sol_all.real + eig_sol_all.imag, 'k')
#plot(alpha_range, abs(eig_sol_all), '--')
#plot(alpha_range, root_all_sol.imag,'k',linewidth=2)
#plot(alpha_range, abs(root_all_sol),'k',linewidth=2)
print(eig(P)[0].imag)
plot(alpha_range, [root_2nd_robust_mag(alpha,gamma) for alpha in alpha_range],'g')



