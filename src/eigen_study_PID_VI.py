#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  1 17:35:41 2020

@author: sologen
"""

import numpy as np
from numpy import *
from numpy.linalg import eig
from matplotlib.pyplot import *

from scipy.optimize import minimize, basinhopping, shgo

import pickle

def calc_root_D(lam, k): # Only k_p term
    
    return roots([1, -(discount*lam + k), k])

def calc_root_PD(lam, kp, kd): # Both P and D terms
    return roots([1, -(1 + kd - kp*(1-discount*lam)), kd])


def calc_root_I(lam, kI, alpha = 0.1, beta = 0.9): # Only k_I term
    return roots([1, -( (1+beta) - (1 + alpha*kI)*(1-discount*lam)), beta*discount*lam])

def calc_root_PI(lam, kp, kI, alpha = 0.05, beta = 0.95):
   return roots([1, -( (1+beta) - (kp + alpha*kI)*(1-discount*lam)), beta*(1-kp*(1-discount*lam))])

def calc_root_PID(lam, kp, kI, kd, alpha = 1., beta = 0.0):

    # print(kp, kI, kd, alpha, beta)
    # kI = 0
    return roots([1, -( (1+kd+beta) - (kp + alpha*kI)*(1-discount*lam)),
                beta*(1+kd) + kd - beta*kp*(1-discount*lam), -kd*beta])


def kPD_analytic(disc):

    kappa = lambda disc: (1+disc)/(1-disc)
    kp = 2/(1 + sqrt(1-disc**2))
    kd = ( (sqrt(kappa(disc)) - 1)/(sqrt(kappa(disc)) + 1) )**2

    return array([kp,kd])



def comp_roots_ksweep(lam, k_range = None, controller = None):
    
    if controller is None:
        controller = calc_root_D

    if k_range is None:
        k_range = linspace(0.,1,100)
    
#    rts = lambda k: roots([1, -(discount*lam + k), k])
    root_lst = array([controller(lam, k) for k in k_range])

    return root_lst, k_range



def comp_roots_thetasweep(mag,k, controller = None):

    if controller is None:
        controller = calc_root_D

    theta_range = linspace(0.,2*pi,200)
    # rts = lambda lam: roots([1, -(discount*lam + k), k])
    root_lst = np.array([controller(mag*exp(theta*1j)) for theta in theta_range])

    return root_lst, theta_range


def gain_sweep_study(lamb, cont_type = 'D'):

    if cont_type == 'D':
        controller = calc_root_D
    elif cont_type == 'I':
        controller = calc_root_I
    
    k_range = linspace(-1,1., 200)
    root_lst, k_range = comp_roots_ksweep(lamb, k_range = k_range, controller = controller)
    # root_lst_2, k_range = comp_roots_ksweep(0.8*exp(0.5*pi/2*1j), controller = calc_root_I)

    # kPD_opt = kPD_analytic(lamb)
    # root_opt = calc_root_PD(lamb, kPD_opt[0],kPD_opt[1])
    # print(kPD_opt)
    # print(root_opt)

    x_max = max(1.1, np.max(root_lst.real) + 0.1 )
    x_min = min(-1.1, np.min(root_lst.real) - 0.1 )

    y_max = max(1.1, np.max(root_lst.imag) + 0.1 )
    y_min = min(-1.1, np.min(root_lst.imag) - 0.1)


    # plot(k_range, root_lst,'b')
    #plot(k_range, root_lst_2,'k')

    fig, (ax1,ax2) = subplots(2,1) #, sharey=True
    ax1.plot(k_range, abs(root_lst),'b')
    # ax1.plot(k_range, abs(root_lst_2),'r')
    # ax1.plot(k_range, len(k_range)*[1.],':')

    # figure()
    ax2.set_xlim(x_min,x_max)
    ax2.set_ylim(y_min,y_max)

    grid("on")
    axis("on")
    xlabel('Real', fontsize = 20)
    ylabel('Imaginary', fontsize=20)

    # ax2.set_xlim(-2.1,+2.1)
    # ax2.set_ylim(-2.1,+2.1)
    ax2.add_artist( Circle( (0,0),1., fill=False, linewidth=2) )

    radius_range = [0.25, 0.5, 0.75]
    [ax2.add_artist( Circle( (0,0),rad, linestyle = ':', fill=False) ) \
     for rad in radius_range]

    # ax2.plot(root_lst.real, root_lst.imag,'b.')
    # ax2.plot(root_lst_2.real, root_lst_2.imag,'r')

    ax2.plot(root_lst[0].real, root_lst[0].imag,'rx',
                markeredgewidth=2, markersize=15)

    ax2.plot(root_lst[-1].real, root_lst[-1].imag,'ro',
                markeredgewidth=2, fillstyle = 'none', markersize=15)
    ax2.plot(root_lst[:].real, root_lst[:].imag,'b.', markersize=1)




def compare_rates_reversible_chain(random_restarts = 10, fig_filename = None):


    disc_range = hstack([linspace(0,0.65,10), linspace(0.7,0.999,20)])
    # disc_range = linspace(0.7,0.999,10)
    # disc_range = logspace(log(0.99), log(0.99999), 10)
    # disc_range = linspace(0.9999,0.9,10)

    # disc_range =np.array([0.9,0.95,0.99])

    # kd_range = linspace(-1,1.,5)
    # kp_range = [1.0,0.9,1.1]
    # kp_range = linspace(0,2,5)
    # max_root_kdkp = lambda kd,kp,disc: np.max(np.abs(array([calc_root_pd(lam,kd,kp) for lam in linspace(-disc,disc,50)] )))
    



    max_root_kD = lambda k,disc, lam_range: np.max(np.abs(array([calc_root_D(lam,k[0]) for lam in lam_range] )))
    max_root_kPD = lambda k,disc, lam_range: np.max(np.abs(array([calc_root_PD(lam,k[0],k[1]) for lam in lam_range] )))
    # max_root_kdkp = lambda k,disc: np.max(np.abs(array([calc_root_pd(lam,k[0],1) for lam in linspace(-disc,disc,20)] )))

    max_root_kI = lambda k,disc, lam_range: np.max(np.abs(array([calc_root_I(lam,k[0]) for lam in lam_range] )))
    max_root_kPI = lambda k,disc, lam_range: np.max(np.abs(array([calc_root_PI(lam,k[0],k[1]) for lam in lam_range] )))
    max_root_kPI_alphabeta = lambda k,disc, lam_range: np.max(np.abs(array([calc_root_PID(lam,k[0],k[1],0,k[2],1 - k[2]) for lam in lam_range] )))
    # max_root_kPID = lambda k,disc: np.max(np.abs(array([calc_root_PID(lam,k[0],k[1],k[2]) for lam in linspace(-disc,disc,20)] )))
    # max_root_kPID = lambda k,disc: np.max(np.abs(array([calc_root_PID(lam,k[0],k[1],k[2], k[3], k[4]) for lam in lam_range] )))
    max_root_kPID = lambda k,disc, lam_range: np.max(np.abs(array([calc_root_PID(lam,k[0],k[1],k[2], k[3], 1-k[3]) for lam in lam_range] )))    

    # XXX CAN BE ERASED XXX
    # kappa = lambda disc: (1+disc)/(1-disc)
    # kPD_analytic = lambda disc: [2/(1 + sqrt(1 -disc**2)),
    #                             ( (sqrt(kappa(disc)) - 1)/(sqrt(kappa(disc)) + 1) )**2]
    # # kPD_analytic_p = lambda disc: 2/(1 + sqrt(1 -disc**2))



    theta_range = linspace(0.,2*pi,50)

    
    effective_disc = []
    effective_disc_opt = []

    effective_disc_optimal_choice = []

    optimized_param = []

    # x0 = [0.00,1.]
    # x0 = [1.,0.00] # PD or PI (fixed alpha and beta)
    x0 = [1.,0.0, 0.0] #, 0.0] # PI + alpha + beta (=1-alpha)
    # x0 = [1.,0.0, 0.0]
    # x0 = [-0.1]
    # x0 = [1.,0.0, 0.0, 0.95, 0.05] # PID + alpha + beta
    # x0 = [1.,0.0, 0.0, 0.00] # PID + alpha + beta (= 1 - alpha)
    
    # A little bit of perturbation helps!
    x0 += 0.01*np.random.randn(len(x0))

    # max_root = max_root_kPD
    max_root = max_root_kPI_alphabeta
    # max_root = max_root_kPID

    # bnd = ((0,2), (-2,2), (-2,2), (-1,1)) #, (-1,1)) # For PID + alpha/beta
    bnd = ((-10,10), (-10,10), (-1,1)) # For PI + alpha/beta
    # bnd = ((0,1), (-3,3), (-1,1)) # For PI + alpha/beta (restrictive)
    # bnd = ((0,2), (-1,1) ) # For PD

    # bnd = None


    for disc in disc_range:

        lam_range = linspace(-disc,1*disc,2)
        lam_range_extended = linspace(-disc,1*disc,50)
        # lam_range = [disc*(1+exp(1j*theta))/sqrt(2) for theta in theta_range]
        # lam_range = [disc*(0.7 + 0.3*(exp(1j*theta))) for theta in theta_range]



        # print(x0)
        # x0 = [1.,0.0, 0.0, 0.0] # PID + alpha + beta
        # res = minimize(max_root_kdkp,x0,disc,method='powell')
        # res = minimize(max_root_kPD,x0,disc,method='powell')
        # res = minimize(max_root_kPI,x0,disc, method='Nelder-Mead')
        # bnd = ((0,2), (-2,2), (0,1), (0,1)) # For PI + alpha/beta
        # res = minimize(max_root_kPI_alphabeta,x0,disc,bounds = bnd) #, method='Powell')

        minimizer_kwargs = {"args":disc}
        # res = basinhopping(max_root_kPI_alphabeta,x0, minimizer_kwargs = minimizer_kwargs)

        # bounds = [(0,2), (-2, 2), (-1,1), (-1, 1)]
        # res = shgo(max_root_kPI_alphabeta,bounds)

        # res = minimize(max_root_kPI_alphabeta,x0,disc,bounds = None, method='Powell')

    
        best_eval = 1e5
        best_res = []
        for ii in range(random_restarts):
            x0_init = x0 + 0.01*np.random.randn(len(x0))
            # res = minimize(max_root_kPI_alphabeta,x0_init,disc, method = 'L-BFGS-B',  bounds = bnd) # method='L-BFGS-B')
            res = minimize(max_root,x0_init,(disc, lam_range), method = 'L-BFGS-B',  bounds = bnd) # method='L-BFGS-B')
            if res.fun < best_eval:
                # print('Found a better one!')
                best_eval = res.fun
                best_res = res

                x0 = best_res.x # XXX Experimental
                # print(ii)
    
        res = best_res

        effective_disc_opt.append(res.fun)

        # res = minimize(max_root_kPI_alphabeta,x0,disc, method = 'L-BFGS-B',  bounds = bnd) # method='L-BFGS-B')
        # res = basinhopping(max_root_kPID,x0, minimizer_kwargs = minimizer_kwargs)

    
        eff_disc = max_root(res.x, disc, lam_range_extended)

        effective_disc.append(eff_disc)

        optimized_param.append(res.x)

        x0 = res.x # + 0.01*np.random.randn(len(x0))

        # print(res)
        

        # kd_analytic = (2 - disc) - 2*sqrt(1- disc)
        # kappa = (1+disc)/(1-disc)
        # kd_kp_analytic = ( (sqrt(kappa) - 1)/(sqrt(kappa) + 1) )**2

        # print('Disc:', disc, '\nNum opt:', x0)
        # print('Disc:', disc, '\nNum opt:', x0, '\nAnalytical:', kd_analytic)

        # print('Disc:', disc, '\nNum opt:', res.x, '\nAnalytical:', kPD_analytic(disc))
        print(disc)

    # effective_disc = [min([max_root_kdkp(kd,kp, disc) for kd in kd_range for kp in kp_range]) for disc in disc_range]

        effective_disc_optimal_choice.append(max_root_kPD(kPD_analytic(disc),disc, lam_range_extended))

        # XXX TEMP
        # k_p = 1./(2*(1+disc))
        # k_I = k_p
        # effective_disc_optimal_choice.append(max_root([k_p, k_I, 1.],disc, lam_range_extended))
    
    optimized_param = array(optimized_param)
    kPD_opt_analytic = array([kPD_analytic(disc) for disc in disc_range])
    
    figure()
    subplot(2,1,1)
    subplots_adjust(hspace = 0.4)
    plot(disc_range, disc_range, linewidth = 2, label='Conventional VI')
    plot(disc_range, effective_disc,'r',linewidth=2,label='PI (Numerical Opt)');
    # plot(disc_range, effective_disc_opt,'r--',linewidth=2,label='Numerical Opt (optimizer persp)');
    
    # plot(disc_range, sqrt(disc_range))

    plot(disc_range, effective_disc_optimal_choice, 'g', linewidth=3, label='PD Optimal (Analytic)')

    effective_disc_PD_analyt = sqrt(kPD_analytic(disc_range)[1])
    # plot(disc_range, effective_disc_PD_analyt,'m--', label='PD Optimal (effective)')

    Goyal_AVI = lambda disc: 1 - np.sqrt((1.-disc)/(1+disc))
    # Goyal_MVI = lambda disc: (np.sqrt(1-disc) - np.sqrt(1+disc) )/(np.sqrt(1-disc) + np.sqrt(1+disc) )
    Goyal_MVI = lambda disc: disc/(1 + np.sqrt(1 - disc**2))

    effective_disc_Goyal_AVI = Goyal_AVI(disc_range)
    effective_disc_Goyal_MVI = Goyal_MVI(disc_range)

    # plot(disc_range, effective_disc_Goyal_AVI,'k')
    # plot(disc_range, effective_disc_Goyal_MVI,'k--')

    legend(fontsize=10)
    xlabel('Discount factor', fontsize=20)
    ylabel('Effective discount factor', fontsize=20)
    # axis("tight")
    grid(True,which='both')
    tick_params(labelsize=15)



    # figure()
    subplot(2,1,2)
    plot(disc_range,optimized_param[:,0], linewidth = 2, label='PI: k_p (numerical)')
    plot(disc_range,optimized_param[:,1], linewidth = 2, label='PI: k_I (numerical)')
    # plot(disc_range,optimized_param[:,2], label='k_P (Num Opt)')
    plot(disc_range,optimized_param[:,2], linewidth = 2, label='PI: alpha (numerical)')

    plot(disc_range, kPD_opt_analytic[:,0], '--',linewidth = 2, label='PD: k_p (analytical)')
    plot(disc_range, kPD_opt_analytic[:,1], '--',linewidth = 2, label='PD: k_d (analytical)')

    legend(fontsize=10)
    xlabel('Discount factor', fontsize=20)
    ylabel('Gain parameters', fontsize=20)
    axis("tight")
    grid(True,which='both')
    tick_params(labelsize=15)

    # Saving the figure
    if fig_filename is not None:
        savefig('figures/'+fig_filename+'.pdf')

        # Pickle the data, as this is an expensive computation
        with open('data/'+fig_filename+'.pickle', 'wb') as f:
            print("Pickling the data ... ")
            data_pickle = {'optimized_param': optimized_param,
                           'kPD_opt_analytic': kPD_opt_analytic,
                            'disc_range': disc_range,
                            'bnd': bnd
                            }

            pickle.dump(data_pickle, f)



    return optimized_param


# def empirical


def study_PI(disc = 0.9):

    lam_range = linspace(-disc*1.,disc,20)
    max_root_kI = lambda k,disc: np.max(np.abs(array([calc_root_I(lam,k[0]) for lam in lam_range] )))

    kI_range = linspace(-5,5)

    max_rt_I = []
    for kI in kI_range:
        max_rt_I.append(max_root_kI([kI], disc))

    plot(kI_range, max_rt_I)
    print(min(max_rt_I))
    # return max_rt_I
    

def lambda_sweep_study(disc):


    # theta_range = linspace(0.,2*pi,200)
    # # rts = lambda lam: roots([1, -(discount*lam + k), k])
    # root_lst = np.array([controller(mag*exp(theta*1j)) for theta in theta_range])

    # lam_range = linspace(-disc,disc,20)

    theta_range = linspace(0.,2*pi,50)
    lam_range = [disc*exp(1j*theta) for theta in theta_range]

    max_root_kPD = lambda k,disc:\
                    np.max(np.abs(array([calc_root_PD(lam,k[0],k[1]) for lam in lam_range] )))

    max_root_kI = lambda k,disc:\
                    np.max(np.abs(array([calc_root_I(lam,k[0]) for lam in lam_range] )))

    max_root_kPI = lambda k,disc:\
                    np.max(np.abs(array([calc_root_PI(lam,k[0],k[1]) for lam in lam_range] )))


    kPD_opt = kPD_analytic(disc)
    root_opt = calc_root_PD(disc, kPD_opt[0],kPD_opt[1])
    print('kPD_opt:', kPD_opt)
    print('root_opt:', root_opt)

    k_range = linspace(-0.2,0.2,200)

    max_root = []
    for k in k_range:
        # max_root.append(max_root_kPD([kPD_opt[0],k],disc))
        max_root.append(max_root_kPI([1,k],disc))

    plot(k_range, max_root)
    print(min(max_root))



discount = 1.
#lam = 0.97
#k = 0.2

# pol = [1, -(discount*lam + k), k]

# lambda lam: roots([1, -(discount*lam + k), k])
# print( roots(pol) )

if False:
    lam1 = 0.99
    lam2 = 0.9
    k1 = -(lam1 - 2) + 2*sqrt(1 - lam1)
    k2 = -(lam1 - 2) - 2*sqrt(1 - lam1)
    print("k1, k2 = ", k1, k2)
    print("expected eigenvalue:", 1 - sqrt(1-lam1))
    
    Num = (2 - lam1 - 2*sqrt(1 - lam1)) - (1 - sqrt(1 - lam1))*abs(lam2)
    Den = 2 - sqrt(1 - lam1)
    k_2_stable = Num/Den
    print('K condition: k < ', k_2_stable)
    
    
    root_theta_1, theta_range = comp_roots_thetasweep(lam1,k2)
    root_theta_2, theta_range = comp_roots_thetasweep(lam2,k2)
    
    fig, (ax1) = subplots(1,1) #, sharey=True
    ax1.add_artist( Circle( (0,0),1., fill=False) )
    ax1.set_xlim(-2.1,+2.1)
    ax1.set_ylim(-2.1,+2.1)
    ax1.plot(root_theta_1.real, root_theta_1.imag,'b.')
    ax1.plot(root_theta_2.real, root_theta_2.imag,'r.')
    print (np.max(abs(root_theta_1)), np.max(abs(root_theta_2)))
    
    figure()
    plot(theta_range, abs(root_theta_1))
    plot(theta_range, abs(root_theta_2),'--')

#print('maximizer theta:', theta_range[argmax(abs(root_theta_2), axis = 0)] )

#lam1 = 0.98

if False:
    cont_type = 'D'

    root_lst_1, k_range = comp_roots_ksweep(-0.5, controller = calc_root_D)
    root_lst_2, k_range = comp_roots_ksweep(0.8*exp(0.5*pi/2*1j), controller = calc_root_I)

    #plot(k_range, root_lst_1,'b')
    #plot(k_range, root_lst_2,'k')

    fig, (ax1,ax2) = subplots(2,1) #, sharey=True
    ax1.plot(k_range, abs(root_lst_1),'b')
    ax1.plot(k_range, abs(root_lst_2),'r')
    ax1.plot(k_range, len(k_range)*[1.],':')

    #figure()
    ax2.set_xlim(-2.1,+2.1)
    ax2.set_ylim(-2.1,+2.1)
    ax2.add_artist( Circle( (0,0),1., fill=False) )
    ax2.plot(root_lst_1.real, root_lst_1.imag,'b')
    ax2.plot(root_lst_2.real, root_lst_2.imag,'r')


### Another experiment -- Solving the worst case of quadratic equation for PD
# disc_range = linspace(0.95,1,10)
# k_range = linspace(-1,1.,200)
# max_root_k = lambda k,disc: np.max(np.abs(array([calc_root(lam, k) for lam in linspace(-disc,disc,50)] )))
# effective_disc = [min([max_root_k(k, disc) for k in k_range]) for disc in disc_range]
# figure()
# plot(disc_range, effective_disc,'r');
# plot(disc_range, disc_range)
# plot(disc_range, sqrt(disc_range))

# Goyal_AVI = lambda disc: 1 - np.sqrt((1.-disc)/(1+disc))
# # Goyal_MVI = lambda disc: (np.sqrt(1-disc) - np.sqrt(1+disc) )/(np.sqrt(1-disc) + np.sqrt(1+disc) )
# Goyal_MVI = lambda disc: disc/(1 + np.sqrt(1 - disc**2))

# effective_disc_Goyal_AVI = Goyal_AVI(disc_range)
# effective_disc_Goyal_MVI = Goyal_MVI(disc_range)

# plot(disc_range, effective_disc_Goyal_AVI,'k')
# plot(disc_range, effective_disc_Goyal_MVI,'k--')

random_restarts = 1000
# optimized_param = compare_rates_reversible_chain(random_restarts = random_restarts,
                                                #  fig_filename = 'reversible_bound_c'+str(random_restarts))
# gain_sweep_study(lamb = 0.9, cont_type='I')
# lambda_sweep_study(0.95)

# max_rt_I = study_PI(disc = 0.97)