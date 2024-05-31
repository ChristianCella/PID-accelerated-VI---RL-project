#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 19:16:11 2020

@author: sologen
"""

import numpy as np
from matplotlib.pyplot import *
from matplotlib.colors import DivergingNorm

np.random.seed(1)

from collections import namedtuple

# import pickle

from joblib import Parallel, delayed
import multiprocessing

from ValueIteration import *
import FiniteMDP

# from root_locus_controllers import *

# from eigen_study_PID_VI import kPD_analytic

from experiment_param_study import eval_VI_perf

rcParams['pdf.fonttype'] = 42
rcParams['ps.fonttype'] = 42
rcParams["figure.figsize"] = [9,8]


def measure_error_perf(error):
    return np.mean(np.log(error+1e-20), axis = 1)




def gain_adaptation_hyper_param_selection(MDP, discount, pi, hyper_param_list,
                                init_gain = Gain(1,0,0,0.05,0.95),
                                iter_no = 500,
                                error_norm_ord = np.inf,
                                normalization_flag = 'BE2',
                                visualize = True, fig_filename = None):
                                # visualize_state_error = True,
                                # shown_states = None,
                                # ):
                                


    # Converting the norm to an appropriate label for the Y axis
    if error_norm_ord == np.inf:
        norm_label = '\infty'
    else:
        norm_label = str(error_norm_ord)

    print('hyper_param_list:', hyper_param_list)

    (error_acc, V_trace_acc, gain_trace_acc, Vopt_true) = \
    evaluate_gain_adaptation(MDP, discount, pi, hyper_param_list=hyper_param_list+[(0,0)],
                        init_gain=init_gain,
                        iter_no=iter_no, error_norm_ord=error_norm_ord,
                        gain_adaptation=True, normalization_flag = normalization_flag)

    # The last item is for the conventional VI
    error_conv_VI = error_acc[-1]

    # Finding the best hyper-parameter
    error_acc = np.array(error_acc[:-1])

    perf_acc = measure_error_perf(error_acc)
    hp_best = np.argmin(perf_acc)
    print('hp_best and best parameters', hp_best, hyper_param_list[hp_best])

    # print(np.shape(error_acc))
    # print('Mean:', np.mean(error_acc, axis = 1) )
    # print('Mean (log):', np.mean(np.log(error_acc+1e-20), axis = 1))

    # hp_best = np.argmin(np.mean(np.log(error_acc+1e-20), axis = 1))
    # print(hp_best, hyper_param_list[hp_best])

    if visualize:
    
        line_style_list = ['-', '--', '-.', ':']
        figure()
        semilogy(error_conv_VI, label='VI (conventional)', linewidth = 2)
        
        for ind, (meta_lr, normalization_eps) in enumerate(hyper_param_list):
            # print(gain, ind)

            semilogy(error_acc[ind],
                    # label=r'VI(PID) with hyper-parameter $(\alpha, \epsilon)=$('+
                    label=r'$(\eta, \epsilon)=$('+
                    str(meta_lr)+', '+str(normalization_eps)+')',
                    linewidth = 2,
                    linestyle=line_style_list[ind % len(line_style_list)])

        xlabel('Iteration', fontsize=20)
        if pi is None:
            ylabel('$\||V_k - V^*\||_{'+norm_label+'}$', fontsize = 20)
        else:
            ylabel('$\||V_k - V^{\pi}\||_{'+norm_label+'}$', fontsize = 20)
        legend(fontsize = 10) # 15 for most figures
        axis("tight")
        grid(True,which='both')

        # Saving the figure
        if fig_filename is not None:
            print('Saving ... ')
            savefig('figures/'+fig_filename+'.pdf')



    return hyper_param_list[hp_best], error_acc[hp_best]


def evaluate_gain_adaptation(MDP, discount, pi, hyper_param_list,
                            init_gain = Gain(1,0,0,0.05,0.95),
                            iter_no = 500,
                            error_norm_ord = np.inf,
                            gain_adaptation = True, 
                            normalization_flag = 'BE2',
                            make_parallel = True):
                            # visualize = True, fig_filename = None):
                                # visualize_state_error = True,
                                # shown_states = None,
                                # ):
                                
    # print ('Computing the value functions ...') # XXX REMOVE 

    # We compute the "true" optimal value function by running the conventional VI
    # for 10*iter_no.
    Vopt_true, Qopt_true, V_trace_orig_VI = \
        value_iteration(MDP.R, MDP.P, discount, IterationsNo=10*iter_no, policy=pi)
        
    error_list = []
    V_trace_list = []
    gain_trace_list = []

    if make_parallel:
        cpu_no = multiprocessing.cpu_count()
        print('Parallel running on ', cpu_no, ' CPUs!')

        VI_compact = lambda meta_lr, normalization_eps: value_iteration_with_acceleration_new(
                                            MDP.R, MDP.P, discount, IterationsNo=iter_no,
                                            alpha = 0., accelation_type='PID',
                                            gain = init_gain,
                                            gain_adaptation=True,
                                            meta_lr = meta_lr,
                                            normalization_flag = normalization_flag,
                                            normalization_eps = normalization_eps,
                                            policy = pi)

        # print("hyper_param_list:", hyper_param_list)

        trace_list = Parallel(n_jobs=cpu_no)(delayed(VI_compact)(meta_lr, normalization_eps ) for 
                                        meta_lr, normalization_eps in hyper_param_list)

        for output_trace in trace_list:
            V_trace_new = output_trace[2]
            gain_trace_new = output_trace[-1]

            error_list.append(np.linalg.norm(V_trace_new - Vopt_true, axis = 0, ord=error_norm_ord))
            V_trace_list.append(V_trace_new)
            gain_trace_list.append(gain_trace_new)

    else:
        for (meta_lr, normalization_eps) in hyper_param_list:
            print(meta_lr, normalization_eps)

            if meta_lr is None:
                meta_lr, normalization_eps = 0, 0

            # if gain_adaptation:
            #     meta_lr, normalization_eps = hyper_param
            # else:
            #     meta_lr, normalization_eps = 0, 0

            Vopt_new, Qopt_new, V_trace_new, \
                Q_trace_new, dV_trace_new, dQ_trace_new, z_trace_new, \
                BE_Q_trace_new, BE_Q_integ_trace_new, gain_trace_new = \
                value_iteration_with_acceleration_new(MDP.R, MDP.P, discount, IterationsNo=iter_no,
                                                        alpha = 0., accelation_type='PID',
                                                        gain = gain,
                                                        gain_adaptation=gain_adaptation,
                                                        meta_lr = meta_lr, 
                                                        normalization_flag = normalization_flag,
                                                        normalization_eps = normalization_eps,
                                                        policy = pi)

            error_list.append(np.linalg.norm(V_trace_new - Vopt_true, axis = 0, ord=error_norm_ord))
            V_trace_list.append(V_trace_new)
            gain_trace_list.append(gain_trace_new)

        

    return error_list, V_trace_list, gain_trace_list, Vopt_true


def experiment_gain_adaptation_garnet(discount, pi, hyper_param_list,
                                state_size = 100, action_size = 4,
                                GarnetParam = (3,5),
                                runs_no = 2,
                                init_gain = Gain(1,0,0,0.05,0.95),
                                with_hp_model_selection = False,
                                normalization_flag = 'BE2',
                                iter_no = 500,
                                error_norm_ord = np.inf,
                                visualize = True, fig_filename = None):

    normalized_conv_VI_error = []
    normalized_acc_error = []
    normalized_best_error = []

    for run in range(runs_no):

        print('Run:', run)
        # Generate a new MDP
        MDP = FiniteMDP.FiniteMDP(StateSize = state_size, ActionSize = action_size,
                                    ProblemType='garnet', GarnetParam=GarnetParam)
        
        # print(normalization_flag)

        (error_list, V_trace_list, gain_trace_list, Vopt_true) = \
        evaluate_gain_adaptation(MDP, discount, pi, hyper_param_list=hyper_param_list+[(0,0)],
                        iter_no=iter_no, error_norm_ord=error_norm_ord, 
                        gain_adaptation=True, normalization_flag=normalization_flag)

    
        # The last item is for the conventional VI
        error_conv_VI = error_list[-1]

        error_list = np.array(error_list[:-1])
        Vopt_true_norm = np.linalg.norm(Vopt_true,ord=error_norm_ord)
        normalized_acc_error.append(error_list / Vopt_true_norm)
        normalized_conv_VI_error.append( error_conv_VI / Vopt_true_norm)


        # semilogy(error_conv_VI / Vopt_true_norm)

        # Find the best hyperparameter
        if with_hp_model_selection:

            # print(np.shape(error_list))

            perf_acc = measure_error_perf(error_list) #[0:-1])
            print('perf_acc:', perf_acc)
            hp_best = np.argmin(perf_acc)
            error_best = error_list[hp_best]
            print('New!:', hp_best, hyper_param_list[hp_best])

            # This is the straightforward way. But we already do all the
            # necessary computation, so we can choose the best hyperparameter
            # just here.
        
            # hp_best, error_best = \
            # gain_adaptation_hyper_param_selection(MDP, discount, pi,
            #                                 hyper_param_list = hyper_param_list,
            #                                 init_gain = gain,
            #                                 iter_no = iter_no,
            #                                 error_norm_ord = error_norm_ord)
            
            normalized_best_error.append(error_best / Vopt_true_norm)

            # print('Here!', np.linalg.norm(error_best - error_best_new))



        print('||Vopt_true||=',Vopt_true_norm )
        # figure()
        # semilogy(error_list.transpose()/Vopt_true_norm)

    normalized_conv_VI_error = np.array(normalized_conv_VI_error)
    normalized_acc_error = np.array(normalized_acc_error)
    normalized_best_error = np.array(normalized_best_error)

    # Visualization part
    if error_norm_ord == np.inf:
        norm_label = '\infty'
    else:
        norm_label = str(error_norm_ord)

    figure()
    iteration_range = np.arange(normalized_conv_VI_error.shape[-1])

    # print('normalized_conv_VI_error.shape',normalized_conv_VI_error.shape)

    err_mean = np.mean(normalized_conv_VI_error, axis = 0)
    err_stderr = np.std(normalized_conv_VI_error, axis = 0) / np.sqrt(runs_no)
    semilogy(err_mean,label='Conventional VI',linewidth = 2)
    fill_between(x = iteration_range, y1=err_mean - err_stderr, y2=err_mean + err_stderr, alpha = 0.1)

    print('err_stderr:', np.mean(err_stderr))

    line_style_list = ['-', '--', '-.']
    
    # for ind in range(hyper_param_list):
    for hp_ind, (meta_lr, normalization_eps) in enumerate(hyper_param_list):     
        err_mean = np.mean(normalized_acc_error[:,hp_ind,:], axis = 0)
        err_stderr = np.std(normalized_acc_error[:,hp_ind,:], axis = 0) / np.sqrt(runs_no)

        (meta_lr, normalization_eps) = hyper_param_list[hp_ind]
        semilogy(err_mean,
                    label=r'$(\eta, \epsilon)=$('+
                    str(meta_lr)+', '+str(normalization_eps)+')',
                    linewidth = 2,
                    linestyle=line_style_list[hp_ind % len(line_style_list)])
        fill_between(x = iteration_range, y1=err_mean - err_stderr, y2=err_mean + err_stderr, alpha = 0.1)

    # print('with_hp_model_selection',with_hp_model_selection)
    if with_hp_model_selection:
        err_mean = np.mean(normalized_best_error, axis = 0)
        err_stderr = np.std(normalized_best_error, axis = 0) / np.sqrt(runs_no)
        semilogy(err_mean,label='Best',linewidth = 2,color='r')
        fill_between(x = iteration_range, y1=err_mean - err_stderr, y2=err_mean + err_stderr,
                    color='r', alpha = 0.1)


        # semilogy(err_mean + err_stderr,':')
        # semilogy(err_mean - err_stderr,':')

    xlabel('Iteration', fontsize=25)
    if pi is None:
        ylabel('$\||V_k - V^*\||_{'+norm_label+'} / \||V^*\||_{'+norm_label+'}$', fontsize = 25)
    else:
        ylabel('$\||V_k - V^{\pi}\||_{'+norm_label+'} / \||V^{\pi} \||_{'+norm_label+'}$', fontsize = 25)
    legend(fontsize = 12, framealpha=0.5) # 15 for most figures
    axis("tight")
    grid(True,which='both')
    xticks(fontsize = 15)
    yticks(fontsize = 15)


    # Saving the figure
    if fig_filename is not None:
        print('Saving ... ')
        savefig('figures/'+fig_filename+'.pdf')

    return normalized_acc_error




# ProblemType = 'randomwalk'
# state_size = 50 # Only for randomwalk

ProblemType = 'garnet'
state_size = 50 # 30 is used in most experiments. In the paper, it is 50.
action_size = 4
GarnetParam = (3,5) # (3,5) is used in most experiments

if ProblemType == 'randomwalk':
    MDP = FiniteMDP.FiniteMDP(state_size, ProblemType='randomwalk')
    ProblemDetail = ''
elif ProblemType == 'garnet':
    # np.random.seed(10)
    # MDP = FiniteMDP.FiniteMDP(state_size, ProblemType='garnet', GarnetParam=(3,5))    
    MDP = FiniteMDP.FiniteMDP(state_size, ActionSize = 4, ProblemType='garnet', GarnetParam=GarnetParam)
    ProblemDetail = str(GarnetParam)+'(state,action)='+str( (state_size, action_size))



discount = 0.99 # 0.99, 0.998

iter_no = 3000 # 3000 for 0.99, 10000 for 0.998
runs_no = 20 # 100 for more detailed one. For the ICML 2021 paper, I used both 20 (appendix) and 100 (main body).

# Selecting the policy.
# If None, it runs the VI with Bellman optimality (computing greedy policy at each step)
pi = [0]*state_size
# pi = np.random.randint(0,2,state_size)
# pi = None

error_norm_ord = np.inf # or 2 or 1 or np.inf

# This is the normalization method used in the
# gain adaptation mechanism
# The choice is between original or BE2.
# BE2 is that seems to work better, and is reported in the paper.
normalization_flag = 'BE2'

discount_str = str(discount).replace('.','p')

gain = Gain(1.0,0,0,0.05,0.95)

# hyper_param_list = [
#                     # (0.01,1e-6), (0.01,1e-4), (0.01,1e-3),
#                     (0.005,1e-6), (0.005,1e-4), (0.005,1e-3),
#                     # (0.0025,1e-6), (0.0025,1e-4), (0.0025,1e-3), (0.0025,1e-2), (0.0025,1e-1),
#                     (0.001,1e-6), (0.001,1e-4), (0.001,1e-3),
#                     (5e-4,1e-6), (5e-4,1e-4), (5e-4,1e-3),
#                     # (1e-4,1e-6), (1e-4,1e-4), (1e-4,1e-3), (1e-4,1e-2), (1e-4,1e-1),
#                     ]

# # This is suitable if we want to sweep over epsilons
# # Range reported in the ICML 2021 paper for eta: 0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5
# eta = 0.5
# # (BR2) 0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5
# # (original) 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2
# eta_str = str(eta).replace('.','p')
# # hyper_param_list = [(eta,1e-8), (eta,1e-6), (eta,1e-4), (eta,1e-2), (eta, 1e-1)]
# sweep_param_detail = 'eta='+eta_str
# # This is a reasonable set for the BR2 normalization
# hyper_param_list = [(eta,1e-20), (eta,1e-16), (eta,1e-12), (eta,1e-10), (eta,1e-8), (eta,1e-6), (eta,1e-4), (eta,1e-2), (eta, 1e-1)]
# # hyper_param_list = [(eta,1e-16), (eta,1e-8), (eta,1e-6), (eta,1e-4), (eta,1e-2), (eta, 1e-1)] # Just for faster evaluation



# This is suitable if we want to sweep over eta
# Range reported in the ICML 2021 paper for eps: 1e-20, 1e-16, 1e-10, 1e-8
normal_eps = 1e-20
eps_str = str(normal_eps).replace('.','p')
hyper_param_list = [(0.001, normal_eps), (0.005, normal_eps), (0.01, normal_eps),
                    (0.02, normal_eps), (0.05, normal_eps), (0.1, normal_eps),
                    # (0.2, normal_eps), (0.5, normal_eps)
                    ]
sweep_param_detail = 'eps='+eps_str

# This is a reasonable set for the original normalization
# hyper_param_list = [(eta,1e-4), (eta,1e-3), (eta,1e-2), (eta, 1e-1)]


# hyper_param_list = [
#                     # (0.001,1e-8), (0.001,1e-6), (0.001,1e-4), (0.001,1e-2),
#                     # (0.005,1e-8), (0.005,1e-6), (0.005,1e-4), (0.005,1e-2),
#                     (0.01,1e-8), (0.01,1e-6), (0.01,1e-4), (0.01,1e-2),
#                     # (0.05,1e-8), (0.05,1e-6), (0.05,1e-4), (0.05,1e-2),
#                     # (0.1,1e-8), (0.1,1e-6), (0.1,1e-4), (0.1,1e-2),
#                     # (0.2,1e-8), (0.2,1e-6), (0.2,1e-4), (0.2,1e-2),
#                     # (0.5,1e-8), (0.5,1e-6), (0.5,1e-4), (0.5,1e-2),
#                     ]



# Suitable for the normalization that only depends on BR**2
# hyper_param_list = [
#                     # (0.01,1e-16), (0.01,1e-8), (0.01,1e-4),
#                     (0.05,1e-16), (0.05,1e-8), (0.05,1e-4),
#                     (0.1,1e-16), (0.1,1e-8), (0.1,1e-4),
#                     (0.2,1e-16), (0.2,1e-8), (0.2,1e-4),
#                     (0.5,1e-16), (0.5,1e-8), (0.5,1e-4),
#                     ]

# hyper_param_list = [
#                     (0.01,1e-16), (0.01,1e-8), (0.01,1e-4),
#                     (0.05,1e-16), (0.05,1e-8), (0.05,1e-4),
#                     (0.1,1e-16), (0.1,1e-8), (0.1,1e-4),
#                     (0.2,1e-16), (0.2,1e-8), (0.2,1e-4),
#                     (0.5,1e-16), (0.5,1e-8), (0.5,1e-4),
#                     ]


# hyper_param_list = [
#                     # (0.01,1e-16), (0.01,1e-8),
#                     # (0.1,1e-16), (0.1,1e-8),
#                     # (0.2,1e-16), (0.2,1e-8), (0.2, 1e-6), (0.2, 1e-4),
#                     (0.5,1e-16), (0.5,1e-8), (0.5,1e-6), (0.5,1e-4)
#                     ]

# hyper_param_list = [
#                     # (0.01,1e-16), (0.01,1e-8), (0.01, 1e-4),
#                     # (0.05,1e-16), (0.05,1e-8), (0.05, 1e-4),
#                     # (0.1,1e-16), (0.1,1e-8), (0.1, 1e-4),
#                     # (0.2,1e-16), (0.2,1e-8), (0.2, 1e-4),
#                     (0.5,1e-16), (0.5,1e-8), (0.5,1e-4)
#                     ]




file_name_detail = ProblemType + ProblemDetail +\
                    '(discount='+discount_str+',normalizer='+str(normalization_flag)+\
                    ','+sweep_param_detail+')-adaptive gain-varying hyperparam'+\
                    ('(control)' if pi is None else '(PE)')+'-runs_no='+str(runs_no)

# file_name_detail = ProblemType + '(discount='+discount_str+')-adaptive gain-varying hyperparam'+\
#                     ('(control)' if pi is None else '(PE)')+'-runs_no='+str(runs_no)

# hp_best, error_best = \
# gain_adaptation_hyper_param_selection(MDP, discount, pi, hyper_param_list, init_gain = gain,
#                                 iter_no = iter_no,
#                                 error_norm_ord = error_norm_ord,
#                                 normalization_flag = normalization_flag,
#                                 fig_filename = file_name_detail)

# plot(error_best,'k')

# (error_list, V_trace_list, gain_trace_list) =\
# evaluate_gain_adaptation(MDP, discount, pi, hyper_param_list=[hp_best],
#                         iter_no=iter_no, error_norm_ord=error_norm_ord, gain_adaptation=True)


# file_name_detail = ProblemType + '(discount='+discount_str+')-sample behaviour-adaptive gain'+\
                        # ('(control)' if pi is None else '(PE)')

# hyper_param_list = [(0.05, 1e-12), (0.1, 1e-12),(None,None)
#                     ]

# (error_list, V_trace_list, gain_trace_list) = \
# evaluate_gain_adaptation(MDP, discount, pi, hyper_param_list=hyper_param_list,
#                         iter_no=iter_no, error_norm_ord=error_norm_ord, gain_adaptation=True)

# error_list = np.array(error_list)
# figure()
# semilogy(error_list.transpose(),'m')

normalized_error =\
experiment_gain_adaptation_garnet(discount = discount, pi = pi, hyper_param_list = hyper_param_list,
                                state_size = state_size, action_size = action_size,
                                GarnetParam = GarnetParam,
                                runs_no = runs_no,
                                init_gain = Gain(1,0,0,0.05,0.95),
                                with_hp_model_selection = True,
                                normalization_flag=normalization_flag,
                                iter_no = iter_no,
                                error_norm_ord = error_norm_ord,
                                visualize = True, fig_filename = file_name_detail)



# semilogy(np.std(normalized_error[:,:,:], axis = 0).transpose())
# legend(str(hyper_param_list))
