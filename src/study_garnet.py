#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 18:03:17 2020

@author: sologen
"""

import numpy as np
from numpy import *
from matplotlib.pyplot import *

from sklearn.neighbors import KernelDensity

import FiniteMDP


def plot_root_density(roots):

    # X = np.array([[rt.real,rt.imag] for rt in roots])
    X = np.array([[rt.real,rt.imag] for rt in roots])

    print(X.shape)
    
    bw = 0.1

    kde = KernelDensity(bandwidth=bw, kernel='tophat')

    kde.fit(X)

    x_range = linspace(-1.2,1.2,100)
    y_range = linspace(-1.2,1.2,100)
    xv, yv = np.meshgrid(x_range, y_range)
    X_eval = vstack([xv.ravel(), yv.ravel()]).transpose()
    # within_circle = ( sum(X_eval**2, axis = 1) <= 1 )

    # X_eval = X_eval[within_circle,:]
    Y_eval = np.exp( kde.score_samples(X_eval) )
    Y_eval = Y_eval.reshape( (x_range.shape[0],y_range.shape[0]) )

    print(shape(X_eval), shape(Y_eval))

    # plot(X_eval[:,0],X_eval[:,1],'.')

    contourf(x_range, y_range, Y_eval)
    colorbar()

    return kde, X


def plot_root_density_real(roots):
    X = np.array([[rt] for rt in roots])

    print(X.shape)

    x_max = np.max(X)
    x_min = np.min(X)
    
    # x_max = 0.3
    # x_min = 0.0

    # bw = 50*(x_max-x_min)/len(roots)
    # print("bw:",bw)
    
    bw = 0.05
    
    kde = KernelDensity(bandwidth=bw, kernel='tophat')

    kde.fit(X)

    
    # X_eval = linspace(-1.2,1.2,100).reshape(-1,1)
    X_eval = linspace(x_min-0.1,x_max+0.1,100).reshape(-1,1)
    Y_eval = np.exp( kde.score_samples(X_eval) )
    # Y_eval = Y_eval.reshape( (x_range.shape[0],y_range.shape[0]) )

    print(shape(X_eval), shape(Y_eval))

    plot(X_eval, Y_eval)
    # # plot(X_eval[:,0],X_eval[:,1],'.')

    # contourf(x_range, y_range, Y_eval)
    # colorbar()

    return kde



ProblemType = 'garnet'
state_size = 1000 # 30 is used in most experiments
action_size = 1
GarnetParam = (10,5) # (3,5) is used in most experiments

# np.random.seed(1)
MDP = FiniteMDP.FiniteMDP(StateSize = state_size, ActionSize = action_size,
                            ProblemType='garnet', GarnetParam=GarnetParam)

P = matrix(MDP.P[0])

# P = (P + P.T)/2

# D, V = np.linalg.eig(MDP.P)


P_eig, P_eigvec = np.linalg.eig(P.transpose())

eig_one_loc = where(np.abs(P_eig - 1) < 1e-10)[0]
print(eig_one_loc)
# P_dom_eig = P_eigvec[:,eig_one_loc[0]]/sum( P_eigvec[:,eig_one_loc[0]] )
# second_eig_mag = sort(abs(P_eig))[-2]

fig, (ax1) = subplots(1,1) #, sharey=True)

x_max = max(1.1, np.max(P_eig.real) + 0.1 )
x_min = min(-1.1, np.min(P_eig.real) - 0.1 )

y_max = max(1.1, np.max(P_eig.imag) + 0.1 )
y_min = min(-1.1, np.min(P_eig.imag) - 0.1)

ax1.set_xlim(x_min,x_max)
ax1.set_ylim(y_min,y_max)
ax1.add_artist( Circle( (0,0),1., fill=False, linewidth=2) )

# plot(P_eig.real, P_eig.imag,'bo')

kde, X = plot_root_density(P_eig)

figure()
# kde = plot_root_density_real(P_eig.real)
# kde = plot_root_density_real(np.abs(P_eig))

# figure()
# v = zeros((state_size,1)).transpose()
# v = ones((state_size,1)).transpose()
# v /= state_size
# # v[0,1] = 1
# u = np.copy(v)
# for m in range(50):
#     v = v*P
#     if m%5 == 0:
#         plot(v.transpose().real)
#         # pause(0.5)
#     # print(v.transpose())
#     # plot((v.transpose()*(P**m)).transpose())


# # P_dom_eig = P_eigvec[:,0]/sum( P_eigvec[:,0] )

# # plot((u*(P**30)).transpose(),'r');
# plot(P_dom_eig.real,'r')

# iter_range = range(50)
# mix_error = [np.linalg.norm( (u*(P**m)).transpose() - P_dom_eig, ord = 1 ) for m in iter_range]
# mix_error_theory = second_eig_mag**iter_range

# figure()
# semilogy(mix_error)
# semilogy(mix_error_theory)