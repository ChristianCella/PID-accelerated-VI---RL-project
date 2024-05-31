# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
Created on Wed May  6 14:27:10 2020

@author: sologen
"""

import numpy as np
from matplotlib.pyplot import *
from matplotlib.colors import DivergingNorm

from collections import namedtuple

import pickle

from joblib import Parallel, delayed
import multiprocessing

from ValueIteration import *
import FiniteMDP

from root_locus_controllers import *

from eigen_study_PID_VI import kPD_analytic

rcParams['pdf.fonttype'] = 42
rcParams['ps.fonttype'] = 42
# matplotlib.use('pgf')
# rcParams['text.usetex'] = True
rcParams["figure.figsize"] = [9,8]

def eval_VI_perf(MDP, discount, policy = None, iter_no = 500, acceleration_type = None, acc_param_list = None,
                 reporting_iter = None, visualize = False, make_parallel = True,
                 error_norm_ord = np.inf,
                 ):


    if acceleration_type is 'PID':
        print('PID controller') #, acc_param_list)
    else:
        return

    if reporting_iter is None:
        reporting_iter = iter_no-1

    error_trace = []

    print ('Computing the value function using the original VI ...')

    Vopt_true, Qopt_true, V_trace_orig_VI = \
        value_iteration(MDP.R, MDP.P, discount, IterationsNo=10*iter_no, policy=pi)
        
    print ('Computing the value function using the accelerated VI ...')
    Vopt, Qopt, V_trace, dQ_trace, dV_trace = \
        value_iteration_with_acceleration(MDP.R, MDP.P, discount, IterationsNo=iter_no,
                                          alpha = 0.0, policy=pi) 

    # Vopt_true, Qopt_true, V_trace_orig_VI = \
    # value_iteration(MDP.R, MDP.P, discount, IterationsNo=10*iter_no, policy=pi)

    error_orig = np.linalg.norm(V_trace - Vopt_true, axis = 0, ord=error_norm_ord)

    if visualize:
        semilogy(error_orig, label='VI (original)')

    if make_parallel:
        cpu_no = multiprocessing.cpu_count()
        print('Parallel running on ', cpu_no, ' CPUs!')
        VI_compact = lambda gain: value_iteration_with_acceleration_new(
                                    MDP.R, MDP.P, discount, IterationsNo=iter_no,
                                    alpha = 0., accelation_type='PID',
                                    gain = gain, policy = pi)[2]

        # print('Hello!', acc_param_list)
        V_trace_acc_list = Parallel(n_jobs=cpu_no)(delayed(VI_compact)(gain) for gain in acc_param_list)
        # Parallel(n_jobs=2)(delayed()(gain) for gain in acc_param_list)
        # Parallel(n_jobs=2)(delayed(i**2)(i) for i in range(10))
        # Parallel(n_jobs=2)(delayed(sqrt)(i ** 2) for i in range(10))
        
        for indx, gain in enumerate(acc_param_list):
            error_acc = np.linalg.norm(V_trace_acc_list[indx] - Vopt_true, axis = 0, ord=error_norm_ord)
            error_trace.append(error_acc[reporting_iter])

    else: # if make_parallel is False
        for gain in acc_param_list:
            print('gain:', gain)
            Vopt_acc, Qopt_acc, V_trace_acc, \
                Q_trace_acc, dV_trace_acc, dQ_trace_acc, z_trace_acc, \
                BE_Q_trace_acc, BE_Q_integ_trace_acc, gain_trace_new = \
                value_iteration_with_acceleration_new(MDP.R, MDP.P, discount, IterationsNo=iter_no,
                                                        alpha = 0., accelation_type='PID',
                                                        gain = gain,
                                                        policy = pi)

            error_acc = np.linalg.norm(V_trace_acc - Vopt_true, axis = 0, ord=error_norm_ord)    
            
            # print(error_acc[reporting_iter])
            error_trace.append(error_acc[reporting_iter])



            # Important visualization
            if visualize:
                # semilogy(error_acc, label='VI with kP = '+str(gain.kP))
                # semilogy(error_acc, label='VI with kD = '+str(gain.kD))
                semilogy(error_acc, label='VI with kI = '+str(gain.kI))  
                # semilogy(np.linalg.norm(V_trace - Vopt_true, axis = 0, ord=np.inf))
                # semilogy(np.linalg.norm(V_trace_new - Vopt_true, axis = 0, ord=np.inf),'--') 
                xlabel('Iteration')
                ylabel('||V_trace - Vopt_true||')
                # legend(['VI (original)','VI with acceleration'])
                # legend([g.kP for g in acc_param_list])
                legend()
                # legend(str(acc_param_list))


    error_trace = np.array(error_trace)

    return error_trace


def study_stability_region_old(MDP, discount, pi, iter_no):

    # kP_range = np.linspace(0,1.2,3)
    kD_range = np.linspace(0.44,0.46,31)
    kI_range = np.linspace(0.225,0.275,51)
    # gains = [Gain(kP, 0., 0., 0., 0.) for kP in kP_range]
    # gains = [Gain(kP, kD, 0., 0., 0.) for kP in kP_range for kD in kD_range]
    gains = [Gain(1., kD, kI, 0.025, 0.975) for kD in kD_range for kI in kI_range]

    # PID with some strange choice of parameters that work well for some problems
    # I don't understand why!
    # gain = [Gain(1. + kI, kI, kI, 0.05, 0.95) for kI in kI_range]

    # print(gains)
    # print(MDP.P)
    reporting_iter = [50, 100, 200, 300, iter_no - 1]

    error_trace = eval_VI_perf(MDP, discount, policy = pi, iter_no = iter_no, acceleration_type='PID',
                acc_param_list=gains, reporting_iter=reporting_iter, make_parallel=True)

    return error_trace, kD_range, kI_range, gains
    # return 0



def study_stability_region(MDP, discount, pi, iter_no):

    # kP_range = np.linspace(0,1.2,3)
    kD_range = np.linspace(0.44,0.46,31)
    kI_range = np.linspace(0.225,0.275,51)
    # gains = [Gain(kP, 0., 0., 0., 0.) for kP in kP_range]
    # gains = [Gain(kP, kD, 0., 0., 0.) for kP in kP_range for kD in kD_range]
    gains = [Gain(1., kD, kI, 0.025, 0.975) for kD in kD_range for kI in kI_range]

    # PID with some strange choice of parameters that work well for some problems
    # I don't understand why!
    # gain = [Gain(1. + kI, kI, kI, 0.05, 0.95) for kI in kI_range]

    # print(gains)
    # print(MDP.P)
    reporting_iter = [50, 100, 200, 300, iter_no - 1]

    error_trace = eval_VI_perf(MDP, discount, policy = pi, iter_no = iter_no, acceleration_type='PID',
                acc_param_list=gains, reporting_iter=reporting_iter, make_parallel=True)

    return error_trace, kD_range, kI_range, gains



def experiment_1D_param_sweep(MDP, discount, pi, param_name, param_range = (0, 1),
                                gain_default = None, resolution = 10, iter_no = 500,
                                reporting_iter = None, error_norm_ord = np.inf,
                                fig_filename = None):


    if gain_default is None:
        print("Default gain")
        gain_default = Gain(1., 0., 0., 0.05, 0.95)
        # gain_default = Gain(1., 0., 0., 0.1, 0.9)

    gain_dict = {'kP': lambda k: Gain(k, gain_default.kI, gain_default.kD, gain_default.I_alpha, gain_default.I_beta),
                 'kI': lambda k: Gain(gain_default.kP, k, gain_default.kD, gain_default.I_alpha, gain_default.I_beta),
                 'kD': lambda k: Gain(gain_default.kP, gain_default.kI, k, gain_default.I_alpha, gain_default.I_beta)
                }

    if not(param_name in ['kD','kI', 'kP']):
        print("(experiment_1D_param_sweep) Incorrect name of the parameter. It should be one of kD, kI or kP")
        return
    
    k_range = linspace(param_range[0],param_range[1],resolution)

    gain_fn = gain_dict[param_name]
    gain = [gain_fn(k) for k in k_range]

    # print(gain)

    # PID with some strange choice of parameters that work well for some problems
    # I don't understand why!
    # gain = [Gain(1. + kI, kI, kI, 0.05, 0.95) for kI in kI_range]

    if reporting_iter is None:
        reporting_iter = np.int_(np.array([1/4, 1/2, 3/4, 1.-1e-6])*iter_no)
        print(reporting_iter)

    error_trace = eval_VI_perf(MDP, discount, policy = pi, iter_no = iter_no, acceleration_type='PID',
                acc_param_list=gain, reporting_iter=reporting_iter,
                make_parallel=True, error_norm_ord=error_norm_ord)

    figure()

    # Converting the norm to an appropriate label for the Y axis
    if error_norm_ord == np.inf:
        norm_label = '\infty'
    else:
        norm_label = str(error_norm_ord)

    # The label for the X axis
    x_label = {'kD':'$k_D$', 'kP':'$k_P$', 'kI':'$k_I$'}[param_name]

    # plot(reporting_iter,error_trace.transpose())
    semilogy(k_range, error_trace, linewidth = 2)
    xlabel(x_label,fontsize = 35)

    if pi is None:
        ylabel('$\||V_k - V^*\||_{'+norm_label+'}$', fontsize = 25)
    else:
        ylabel('$\||V_k - V^{\pi}\||_{'+norm_label+'}$', fontsize = 25)

    # ylabel('$\||V - V_k\||_{'+norm_label+'}$', fontsize = 20)
    legend(reporting_iter, fontsize=20)
    # semilogy(k_range, error_trace,'kx', markeredgewidth = 0.5)
    semilogy(k_range[::3], error_trace[::3],'kx', markeredgewidth = 0.5)
    axis("tight")
    grid(True,which='both')
    tick_params(labelsize=15)
    
    
    # Saving the figure
    if fig_filename is not None:
        savefig('figures/'+fig_filename+'.pdf')

    

def experiment_2D_param_sweep(MDP, discount, pi,
                                param_name = 'kID', param_range = [(0, 1), (0,1)],
                                gain_default = None, resolution = [10,10], iter_no = 500,
                                reporting_iter = None, error_norm_ord = np.inf,
                                fig_filename = None):


    if gain_default is None:
        print("Default gain")
        gain_default = Gain(1., 0., 0., 0.05, 0.95)

    gain_dict = {'kPI': lambda kP, kI: Gain(kP, kI, gain_default.kD, gain_default.I_alpha, gain_default.I_beta),
                 'kPD': lambda kP, kD: Gain(kP, gain_default.kI, kD, gain_default.I_alpha, gain_default.I_beta),
                 'kID': lambda kI, kD: Gain(gain_default.kP, kI, kD, gain_default.I_alpha, gain_default.I_beta)
                }


    if not(param_name in ['kPI','kPD', 'kID']):
        print("(experiment_2D_param_sweep) Incorrect name of the parameter. It should be one of kPI, kPD, kID.")
        return
    
    k1_range = linspace(param_range[0][0],param_range[0][1],resolution[0])
    k2_range = linspace(param_range[1][0],param_range[1][1],resolution[1])

    gain_fn = gain_dict[param_name]
    gain = [gain_fn(k1,k2) for k1 in k1_range for k2 in k2_range]

    # Adding the conventional VI to the list of gains
    gain.append(Gain(1., 0., 0., 0.05, 0.95))

    # print(gain)
    print("Number of gains to be evaluated:", len(gain))

    # return 

    # PID with some strange choice of parameters that work well for some problems
    # I don't understand why!
    # gain = [Gain(1. + kI, kI, kI, 0.05, 0.95) for kI in kI_range]

    if reporting_iter is None:
        reporting_iter = np.int_(np.array([1/4, 1/2, 3/4, 1.-1e-6])*iter_no)
        reporting_len = len(reporting_iter)
        print(reporting_iter)

    error_trace = eval_VI_perf(MDP, discount, policy = pi, iter_no = iter_no, acceleration_type='PID',
                acc_param_list=gain, reporting_iter=reporting_iter,
                make_parallel=True, error_norm_ord=error_norm_ord)

    error_baseline = error_trace[-1]

    # Pickle the data, as this is an expensive computation
    with open('data/'+fig_filename+'.pickle', 'wb') as f:
        print("Pickling the data ... ")
        data_pickle = {'error_trace': error_trace[:-1],
                       'error_baseline': error_baseline,
                'k1_range': k1_range, 'k2_range': k2_range,
                'param_name': param_name, 'reporting_iter': reporting_iter
                }

        pickle.dump(data_pickle, f)

    visualize_2d(param_name, error_trace[:-1],
                k1_range, k2_range, reporting_iter,
                error_baselines=error_baseline, fig_filename=fig_filename,
                visualize_all_iter = False)

    return 


def visualize_2d(param_name, error_trace, k1_range, k2_range, reporting_iter,
                 error_baselines = None, fig_filename = None,
                 visualize_all_iter = False, visualize_eff_plan_horizon = False):

    
    # The labels for the X and Y axes
    y_label, x_label = {'kPI': ('$k_P$','$k_I$'),
                        'kPD': ('$k_P$','$k_D$'),
                        'kID': ('$k_I$','$k_D$')
                        }[param_name]

    reporting_len = len(reporting_iter)
    error = error_trace.reshape(len(k1_range), len(k2_range), reporting_len)
    # error = error_trace.reshape(len(k2_range), len(k1_range), reporting_len)
    # print('error shape', error.shape)
    
    if error_baselines is None:
        error_baselines = [1.]*reporting_len
        print(error_baselines)
        print('(visualize_2d) Warning: The behaviour is not optimized for when the baseline is not provided.')

    print('error_baselines', error_baselines)


    extent = (k2_range[0],k2_range[-1],k1_range[0],k1_range[-1])
    # imshow(np.log10(error_processed),origin='lower', extent=extent, aspect='auto')

    # print(np.min(error_processed))

    # rel_thresh = 1e0
    # img_val = np.log10(error_processed/error_0)*(error_processed <= rel_thresh*error_0) +\
    #           np.log10(rel_thresh)*(error_processed > rel_thresh*error_0)

    if visualize_all_iter:
        reporting_vis_index = arange(reporting_len)
    else:
        reporting_vis_index = [reporting_len-1]

    # print('reporting_vis_index:', reporting_vis_index)

    # Setting the number of rows and columns of the subplot
    # The rows can be of the larger size.
    col_no = np.floor(np.sqrt( len(reporting_vis_index) ))
    row_no = np.ceil(np.sqrt( len(reporting_vis_index) ))

    # print('col_no, row_no:', col_no, row_no)

    for plot_ind, data_ind in enumerate(reporting_vis_index):
        
        img_val = np.log10(error[:,:,data_ind]/error_baselines[data_ind])

        subplot(row_no,col_no,plot_ind+1)
        subplots_adjust(wspace = 0.4, hspace = 0.4)

        # colormaps that look fine: seismic, Spectral_r, bwr
        # interpolation: The PDF generated by "none" is seen as smoothed
        # by Preview on my computer. It works fine with Acrobat Reader.
        # If "nearest" is used, it looks better on Preview, but still a bit smoothing.
        imshow(img_val, origin='lower', extent=extent, 
                aspect='auto', cmap='bwr', 
                interpolation='nearest',
                norm=DivergingNorm(vcenter = 0, vmax=1)
                )

        colorbar()
        xlabel(x_label, fontsize=20)
        ylabel(y_label, fontsize=20)
        title('Iteration: ' + str(reporting_iter[data_ind]), fontsize = 20)
        tick_params(labelsize=15)
        # title(r'$\log_{10}(\frac{error}{error_{VI}})$', fontsize = 20)


    # Saving the figure
    if fig_filename is not None:
        savefig('figures/'+fig_filename+'.pdf')


    # Computing the effective planning horizon
    if visualize_eff_plan_horizon:
        eff_discount = np.exp( np.log(error[:,:,-1]/error[:,:,0]) / (reporting_iter[-1] - reporting_iter[0]) )
        eff_discount_baseline = np.exp( np.log(error_baselines[-1]/error_baselines[0]) / (reporting_iter[-1] - reporting_iter[0]) )

        # Clipping discount at almost 1. This is to avoid having negative planning horizon.
        eff_discount = np.clip(eff_discount, a_min = 0., a_max = 1 - 1e-10)
        eff_plan_horizon = 1./(1. - eff_discount)

        # print('Min and Max discount:', np.min(eff_discount), np.max(eff_discount))

        eff_plan_horizon_baseline = 1./(1. - eff_discount_baseline)

        # Computing the minimum and maximum planning horizon, to be used by imshow
        # We consider eff_plan_horizon_baseline in the computation of minimum, in case
        # the minimum is not among the accelerated VI parameters.
        min_eff_plan_horizon = min( np.min(eff_plan_horizon), eff_plan_horizon_baseline-1)
        max_eff_plan_horizon = np.max(eff_plan_horizon)

        print('eff_discount_baseline, eff_plan_horizon_baseline:',eff_discount_baseline, eff_plan_horizon_baseline)
        print('min_eff_plan_horizon, max_eff_plan_horizon:', min_eff_plan_horizon, max_eff_plan_horizon)
        # print(eff_discount_baseline, eff_plan_horizon_baseline, min_eff_plan_horizon, max_eff_plan_horizon)

        figure()
        imshow(eff_plan_horizon, origin='lower', extent=extent,
                aspect='auto', cmap='bwr',
                interpolation='nearest', 
                norm=DivergingNorm(vcenter = eff_plan_horizon_baseline,
                                    vmin = min_eff_plan_horizon, vmax=1.5*eff_plan_horizon_baseline) )
        colorbar()
        xlabel(x_label, fontsize=20)
        ylabel(y_label, fontsize=20)
        title('Effective Planning Horizon', fontsize = 20)
        tick_params(labelsize=15)

        # Saving the figure
        if fig_filename is not None:
            savefig('figures/'+fig_filename+'(effective horizon).pdf')

        print(shape(eff_discount))
    
    return  error

def experiment_sample_behaviour(MDP, discount, pi, acc_param_list = None,
                                iter_no = 500,
                                error_norm_ord = np.inf,
                                visualize_state_error = True,
                                shown_states = None,
                                fig_filename = None):

    # print(acc_param_list)

    # Converting the norm to an appropriate for the Y axis label
    if error_norm_ord == np.inf:
        norm_label = '\infty'
    else:
        norm_label = str(error_norm_ord)


    print ('Computing the value functions ...')
    
    Vopt_true, Qopt_true, V_trace_orig_VI = \
        value_iteration(MDP.R, MDP.P, discount, IterationsNo=10*iter_no, policy=pi)
        
    # print ('Computing the value function using the accelerated VI ...')
    # XXX Come back to here and change it to value_iteration_with_acceleration_new XXX
    # Vopt_conv_VI, Qopt_conv_VI, V_trace_conv_VI, dQ_trace_conv_VI, dV_trace_conv_VI = \
    #     value_iteration_with_acceleration(MDP.R, MDP.P, discount, IterationsNo=iter_no,
    #                                         alpha = 0.0, policy=pi) 

    Vopt_conv_VI, Qopt_conv_VI, V_trace_conv_VI,\
        Q_trace_conv_VI, dV_trace_conv_VI, dQ_trace_conv_VI, z_trace_conv_VI,\
        BE_Q_trace_conv_VI, BE_Q_integ_trace_conv_VI, gain_trace_conv_VI = \
        value_iteration_with_acceleration_new(MDP.R, MDP.P, discount, IterationsNo=iter_no,
                                            alpha = 0.0, accelation_type='P',
                                            gain = Gain(1, 0., 0.), 
                                            policy=pi) 

    # print('Norm difference:', np.linalg.norm(V_trace_conv_VI - V_trace_conv_VI_1))

    # This is the error of the conventional VI
    error_conv_VI= np.linalg.norm(V_trace_conv_VI - Vopt_true, axis = 0, ord=error_norm_ord)

    # return 
    error_acc = []
    V_trace_acc = []

    for gain in acc_param_list:
        Vopt_new, Qopt_new, V_trace_new, \
            Q_trace_new, dV_trace_new, dQ_trace_new, z_trace_new, \
            BE_Q_trace_new, BE_Q_integ_trace_new, gain_trace_new = \
            value_iteration_with_acceleration_new(MDP.R, MDP.P, discount, IterationsNo=iter_no,
                                                    alpha = 0., accelation_type='PID',
                                                    gain = gain,
                                                    policy = pi)

        error_acc.append(np.linalg.norm(V_trace_new - Vopt_true, axis = 0, ord=error_norm_ord))
        V_trace_acc.append(V_trace_new)

    figure()
    # rcParams['font.size'] = '15'
    line_style_list = ['-', '--', '-.', ':']
    semilogy(error_conv_VI, label='VI (conventional)', linewidth = 2)
    for ind, gain in enumerate(acc_param_list):
        # print(gain, ind)

        semilogy(error_acc[ind],
                label='VI(PID) with $(k_p, k_I, k_d)=$('+
                str(gain.kP)+', '+str(gain.kI)+', '+str(gain.kD)+')',
                # str("{:.2f}".format(gain.kP))+', '+str("{:.2f}".format(gain.kI))+', '+str("{:.2f}".format(gain.kD))+')',
                linewidth = 2,
                linestyle=line_style_list[ind % len(line_style_list)])

    # semilogy(error_acc[ind],'--')
    xlabel('Iteration', fontsize=25)
    # ylabel('||V_trace - Vopt_true||', fontsize = 20)
    if pi is None:
        ylabel('$\||V_k - V^*\||_{'+norm_label+'}$', fontsize = 25)
    else:
        ylabel('$\||V_k - V^{\pi}\||_{'+norm_label+'}$', fontsize = 25)
    # legend(['VI (original)','VI with acceleration'])
    legend(fontsize = 18, framealpha=0.5)
    # axis("tight")
    grid(True,which='both')
    xticks(fontsize = 15)
    yticks(fontsize = 15)

    # Saving the figure
    if fig_filename is not None:
        savefig('figures/'+fig_filename+'(error norm).pdf')




    # Setting the number of rows and columns of the subplot
    # The rows can be of the larger size.
    col_no = int(np.floor(np.sqrt( len(gain_list)+1 )))
    row_no = int(np.ceil(np.sqrt( len(gain_list)+1 )))

    # For debugging purposes. Making sure that I get the right dimensions.
    # print('V_trace_conv_VI:', V_trace_conv_VI.shape)
    # print('V_trace_conv_VI[10:15,:]', V_trace_conv_VI[10:15,:].shape)
    # print('Vopt_true', Vopt_true.shape)

    if pi is None:
        y_label = '$V_k(x) - V^*(x)$'
    else:
        y_label = '$V_k(x) - V^{\pi}(x)$'


    state_no = Vopt_true.shape[0]
    if shown_states is None:
        state_ind = arange(0,state_no)
    elif shown_states=='adaptive':
        state_ind = arange(0,state_no, max(int(state_no/5),1) )
    else:
        state_ind = shown_states
    # Otherwise, it is assumed to be a list of states

    # print('state_ind',state_ind)
    
    # fig, ax = subplots(row_no, col_no)

    # figure()
    # plot(Vopt_true)

    figure()
    
    subplot(row_no, col_no,1)
    subplots_adjust(wspace = 0.3, hspace = 0.5)
    # plot(V_trace_conv_VI.transpose() - Vopt_true[:,0])    
    plot(V_trace_conv_VI[state_ind,:].transpose() - Vopt_true[state_ind].squeeze(), linewidth = 1)
    title('VI (Conventional)', fontsize=10)
    xlabel('Iteration', fontsize=10)
    ylabel(y_label, fontsize=10)
    grid(True,which='both')

    # xticks(fontsize = 10)
    # yticks(fontsize = 10)


    # Only show the legend if the number of states in state_ind isn't too large.
    # The choice of 5 is arbitrary here, and possibly can become a parameter.
    if len(state_ind) <= 5:
        legend(labels=state_ind, loc = 'upper right', fontsize = 5)


    for ind, gain in enumerate(acc_param_list):
        
        # print(gain, ind)

        subplot(row_no, col_no, ind+2)
        # subplots_adjust(wspace = 0.3, hspace = 0.4)
        plot(V_trace_acc[ind][state_ind,:].transpose() - Vopt_true[state_ind].squeeze(), linewidth = 1)
            # label=state_ind)

        title('VI(PID) with $(k_p, k_I, k_d)=$('+\
                str(gain.kP)+', '+str(gain.kI)+', '+str(gain.kD)+')', fontsize=10)

        xlabel('Iteration', fontsize=10)
        ylabel(y_label, fontsize=10)
        
        # Only show the legend if the number of states in state_ind isn't too large.
        # The choice of 5 is arbitrary here, and possibly can become a parameter.
        if len(state_ind) <= 5:
            legend(labels=state_ind, loc = 'upper right', fontsize = 5)

        grid(True,which='both')

    # Saving the figure
    if fig_filename is not None:
        savefig('figures/'+fig_filename+'(state errors).pdf')


    return



def experiment_sample_behaviour_gain_adaptation(MDP, discount, pi, acc_param_list = None,
                                iter_no = 500,
                                error_norm_ord = np.inf,
                                visualize_state_error = True,
                                shown_states = None,
                                meta_lr = None, 
                                normalization_eps = None, normalization_flag = 'BE2',
                                fig_filename = None):

    # print(acc_param_list)

    # Converting the norm to an appropriate for the Y axis label
    if error_norm_ord == np.inf:
        norm_label = '\infty'
    else:
        norm_label = str(error_norm_ord)

    print ('Computing the value functions ...')
    
    Vopt_true, Qopt_true, V_trace_orig_VI = \
        value_iteration(MDP.R, MDP.P, discount, IterationsNo=10*iter_no, policy=pi)
        

    Vopt_conv_VI, Qopt_conv_VI, V_trace_conv_VI,\
        Q_trace_conv_VI, dV_trace_conv_VI, dQ_trace_conv_VI, z_trace_conv_VI,\
        BE_Q_trace_conv_VI, BE_Q_integ_trace_conv_VI, gain_trace_conv_VI = \
        value_iteration_with_acceleration_new(MDP.R, MDP.P, discount, IterationsNo=iter_no,
                                            alpha = 0.0, accelation_type='P',
                                            gain = Gain(1, 0, 0),
                                            policy=pi) 

    # print('Norm difference:', np.linalg.norm(V_trace_conv_VI - V_trace_conv_VI_1))

    # This is the error of the conventional VI
    error_conv_VI= np.linalg.norm(V_trace_conv_VI - Vopt_true, axis = 0, ord=error_norm_ord)

    # return 
    error_acc = []
    V_trace_acc = []
    gain_trace_acc = []

    for gain in acc_param_list:
        Vopt_new, Qopt_new, V_trace_new, \
            Q_trace_new, dV_trace_new, dQ_trace_new, z_trace_new, \
            BE_Q_trace_new, BE_Q_integ_trace_new, gain_trace_new = \
            value_iteration_with_acceleration_new(MDP.R, MDP.P, discount, IterationsNo=iter_no,
                                                    alpha = 0., accelation_type='PID',
                                                    gain = gain,
                                                    gain_adaptation=True,
                                                    meta_lr = meta_lr, 
                                                    normalization_eps = normalization_eps,
                                                    normalization_flag = normalization_flag,
                                                    policy = pi)

        error_acc.append(np.linalg.norm(V_trace_new - Vopt_true, axis = 0, ord=error_norm_ord))
        V_trace_acc.append(V_trace_new)
        gain_trace_acc.append(gain_trace_new)


    line_style_list = ['-', '--', '-.', ':']
    figure()
    semilogy(error_conv_VI, label='VI (conventional)', linewidth = 2)
    for ind, gain in enumerate(acc_param_list):
        # print(gain, ind)

        semilogy(error_acc[ind],
                label='VI(PID) with initial $(k_p, k_I, k_d)=$('+
                str(gain.kP)+', '+str(gain.kI)+', '+str(gain.kD)+')',
                linewidth = 2,
                linestyle=line_style_list[ind % len(line_style_list)])

    # semilogy(error_acc[ind],'--')
    xlabel('Iteration', fontsize=25)
    # ylabel('||V_trace - Vopt_true||', fontsize = 20)
    if pi is None:
        ylabel('$\||V_k - V^*\||_{'+norm_label+'}$', fontsize = 25)
    else:
        ylabel('$\||V_k - V^{\pi}\||_{'+norm_label+'}$', fontsize = 25)
    # legend(['VI (original)','VI with acceleration'])
    legend(fontsize = 15, loc='upper right')
    # axis("tight")
    grid(True,which='both')
    xticks(fontsize = 15)
    yticks(fontsize = 15)


    # Saving the figure
    if fig_filename is not None:
        savefig('figures/'+fig_filename+'(error norm).pdf')


    # Visualizing the gains
    figure()
    for ind, gain in enumerate(acc_param_list):
        # print(gain, ind)

        gain_kP_trace = [gain.kP for gain in gain_trace_acc[ind]]
        gain_kI_trace = [gain.kI for gain in gain_trace_acc[ind]]
        gain_kD_trace = [gain.kD for gain in gain_trace_acc[ind]]

        plot(gain_kP_trace, 'b', linewidth = 2,
                            # label = '$k_p$ with init value:'+str(gain_kP_trace[0]),
                            linestyle=line_style_list[ind % len(line_style_list)])
        plot(gain_kI_trace, 'r', linewidth = 2,
                            # label = '$k_I$',
                            linestyle=line_style_list[ind % len(line_style_list)])
        plot(gain_kD_trace, 'k', linewidth = 2,
                            # label = '$k_d$',
                            linestyle=line_style_list[ind % len(line_style_list)])

        # semilogy(error_acc[ind],
        #         label='VI(PID) with initial $(k_p, k_I, k_d)=$('+
        #         str(gain.kP)+', '+str(gain.kI)+', '+str(gain.kD)+')',
        #         linewidth = 2,
        #         linestyle=line_style_list[ind % len(line_style_list)])

    # semilogy(error_acc[ind],'--')
    xlabel('Iteration', fontsize = 25)
    ylabel('Controller gains', fontsize = 25)

    # # ylabel('||V_trace - Vopt_true||', fontsize = 20)
    # if pi is None:
    #     ylabel('$\||V_k - V^*\||_{'+norm_label+'}$', fontsize = 20)
    # else:
    #     ylabel('$\||V_k - V^{\pi}\||_{'+norm_label+'}$', fontsize = 20)
    # # legend(['VI (original)','VI with acceleration'])
    legend(['$k_p$', '$k_I$', '$k_d$'], fontsize = 20)
    # axis("tight")
    grid(True,which='both')
    xticks(fontsize = 15)
    yticks(fontsize = 15)



    # Saving the figure
    if fig_filename is not None:
        savefig('figures/'+fig_filename+'(gains).pdf')


    # Setting the number of rows and columns of the subplot
    # The rows can be of the larger size.
    col_no = int(np.floor(np.sqrt( len(gain_list)+2 )))
    row_no = int(np.ceil(np.sqrt( len(gain_list)+2 )))


    print('col_no, row_no:', col_no, row_no)
    # For debugging purposes. Making sure that I get the right dimensions.
    # print('V_trace_conv_VI:', V_trace_conv_VI.shape)
    # print('V_trace_conv_VI[10:15,:]', V_trace_conv_VI[10:15,:].shape)
    # print('Vopt_true', Vopt_true.shape)

    if pi is None:
        y_label = '$V_k(x) - V^*(x)$'
    else:
        y_label = '$V_k(x) - V^{\pi}(x)$'


    state_no = Vopt_true.shape[0]
    if shown_states is None:
        state_ind = arange(0,state_no)
    elif shown_states=='adaptive':
        state_ind = arange(0,state_no, max(int(state_no/5),1) )
    else:
        state_ind = shown_states
    # Otherwise, it is assumed to be a list of states

    # print('state_ind',state_ind)
    
    # fig, ax = subplots(row_no, col_no)

    # figure()
    # plot(Vopt_true)

    figure()
    
    subplot(row_no, col_no,1)
    subplots_adjust(wspace = 0.3, hspace = 0.5)
    # plot(V_trace_conv_VI.transpose() - Vopt_true[:,0])    
    plot(V_trace_conv_VI[state_ind,:].transpose() - Vopt_true[state_ind].squeeze(), linewidth = 1)
    title('VI (Conventional)', fontsize=10)
    xlabel('Iteration', fontsize=10)
    ylabel(y_label, fontsize=10)
    grid(True,which='both')

    # Only show the legend if the number of states in state_ind isn't too large.
    # The choice of 5 is arbitrary here, and possibly can become a parameter.
    if len(state_ind) <= 5:
        legend(labels=state_ind, loc = 'upper right', fontsize = 5)


    for ind, gain in enumerate(acc_param_list):
        
        # print(gain, ind)

        subplot(row_no, col_no, ind+2)
        # subplots_adjust(wspace = 0.3, hspace = 0.4)
        plot(V_trace_acc[ind][state_ind,:].transpose() - Vopt_true[state_ind].squeeze(), linewidth = 1)
            # label=state_ind)

        title('VI(PID) with initial $(k_p, k_I, k_d)=$('+\
                str(gain.kP)+', '+str(gain.kI)+', '+str(gain.kD)+')', fontsize=10)

        xlabel('Iteration', fontsize=10)
        ylabel(y_label, fontsize=10)
        
        # Only show the legend if the number of states in state_ind isn't too large.
        # The choice of 5 is arbitrary here, and possibly can become a parameter.
        if len(state_ind) <= 5:
            legend(labels=state_ind, loc = 'upper right', fontsize = 5)

        grid(True,which='both')

    # Saving the figure
    if fig_filename is not None:
        savefig('figures/'+fig_filename+'(state errors).pdf')


if __name__ == '__main__':
    
    # We can set what expertiment to run here.
    flag_experiment_1D_param_sweep = True
    flag_experiment_2D_param_sweep = False
    flag_experiment_sample_behaviour = True
    flag_experiment_sample_behaviour_gain_adaptation = True

    flag_experiment_1D_param_sweep_root_locus = False

    ProblemType = 'randomwalk'

    # ProblemType = 'garnet'
    # GarnetParam = (3,5) # (30,5)

    state_size = 50 # Only for randomwalk and garnet
    action_size = 4



    # ProblemType = 'TwoStatesReal'
    # ProblemType = 'ThreeStatesComplex'

    if ProblemType == 'randomwalk':
        MDP = FiniteMDP.FiniteMDP(state_size, ProblemType='randomwalk')
    elif ProblemType == 'garnet':
        # np.random.seed(1) # Set the seed, so we always run the same Garnet. Otherwise, it would be random.
        MDP = FiniteMDP.FiniteMDP(state_size, ActionSize = action_size, ProblemType='garnet', GarnetParam=GarnetParam)    
    elif ProblemType == 'TwoStatesReal':
        MDP = FiniteMDP.FiniteMDP(2,ActionSize = 1, ProblemType='TwoThreeStatesProblem',TwoThreeStatesParam = 0.2)
    elif ProblemType == 'ThreeStatesComplex':
        MDP = FiniteMDP.FiniteMDP(3,ActionSize = 1, ProblemType='TwoThreeStatesProblem', TwoThreeStatesParam = 0.1)

    state_size = np.shape(MDP.P[0])[0]


    # print('Eigenvalues of P[0] are:', eig(MDP.P[0])[0])

    # Problem_Dic = {'randomwalk':  FiniteMDP.FiniteMDP(state_size, ProblemType='randomwalk'),
    #                 'TwoStateReal': FiniteMDP.FiniteMDP(2,ActionSize = 1,
    #                                 ProblemType='TwoThreeStatesProblem',TwoThreeStatesParam = 0.95),
    #                 'ThreeStateComplex': FiniteMDP.FiniteMDP(3,ActionSize = 1, 
    #                                 ProblemType='TwoThreeStatesProblem')}
    # MDP = Problem_Dic[ProblemType]



    # MDP.R = np.random.random(state_size)
        
    #    state_size = 5 #P.shape[0]
    #    P = np.eye(state_size)
    ##    P = random.random( (state_size,state_size) )    
    #    P = np.matrix(P)
    #    
    #    P = P/np.sum(P,axis = 1)
    #    
    #    R = np.zeros(state_size)
    #    R[int(state_size/2)] = 1.
    #    R[0] = -1.
    #
    #    MDP = FiniteMDP.FiniteMDP(state_size, 2)
    #    MDP.P[0] = P
    #    MDP.P[1] = P
    #    MDP.R = R

    discount = 0.99

    # For the gain adaptation experiments, 2000 for 0.99; 10000 for 0.998.
    # For observing sample behaviour, 500 is suitable.
    iter_no = 2000

    # Selecting the policy.
    # If None, it runs the VI with Bellman optimality (computing greedy policy at each step)
    pi = [0]*state_size # Policy that always chooses the first action
    # pi = np.random.randint(0,2,state_size) # Random policy
    # pi = None # For the Bellman optimality


    error_norm_ord = np.inf # or 2 or 1 or np.inf

    discount_str = str(discount).replace('.','p')

    # For studying the stability region
    # Produces a 2D map
    if flag_experiment_2D_param_sweep:

        resolution = [100,100] # resolution of the control gain study

        param_name = 'kID'
        # param_range = [(-0.2, 0.4), (-0.3, 0.3)] # Good range for kID (PE)
        # param_range = [(-1., 4), (-0.3, 0.5)] # Good range for kID (control)
        
        # param_name = 'kPD'
        # param_range = [(0.5, 1.5), (-0.3, 0.5)] # Good range for kPD (PE)
        # param_range = [(0.2, 1.6), (-0.3, 0.8)] # Good range for kPD (control)

        # param_name = 'kPI'
        # param_range = [(0.4, 1.3), (-0.9, 0.3)] # Good range for kPI (PE)
        # param_range = [(0.2, 1.3), (-1, 4)] # Good range for kPI (control) (not settled! XXX)

        
        gain_default = Gain(1., 0., 0., 0.05, 0.95)
        # gain_default = Gain(1., 0., 0.0, 0.3, 0.7)    

        # param_name = 'kP'
        # param_range = (0.8, 1.2)

        file_name_detail = ProblemType +'(discount='+discount_str+')-2D sweep-'+ param_name +\
                            ('(control)' if pi is None else '(PE)')
        if False:
            experiment_2D_param_sweep(MDP, discount, pi, 
                                    param_name = param_name,
                                    param_range = param_range, resolution = resolution,
                                    gain_default = gain_default,
                                    iter_no = iter_no,
                                    error_norm_ord = error_norm_ord, # Note that this is for the sup-norm
                                    fig_filename=file_name_detail)

        if True:
            # with open('data/2d.pickle', 'rb') as f:
            with open('data/'+file_name_detail+'.pickle', 'rb') as f:
                data_pickled = pickle.load(f)

            error = visualize_2d(data_pickled['param_name'], data_pickled['error_trace'],
                        data_pickled['k1_range'], data_pickled['k2_range'],
                        data_pickled['reporting_iter'],
                        visualize_all_iter = False,
                        visualize_eff_plan_horizon = True,
                        error_baselines = data_pickled['error_baseline'],
                        fig_filename = file_name_detail)
    
    

        # error_trace, kD, kI, gains = study_stability_region(MDP, discount, pi, iter_no)

        # # kP_range = np.array([g.kP for g in gains])
        # # kD_range = np.array([g.kD for g in gains])

        # # kP_range = np.linspace(0,1.2,3)
        # # kD_range = np.linspace(0,0.1,4)

        # error = error_trace.reshape(len(kD), len(kI), 5)
        # # error_0 = error[np.where(kD == 0), np.where(kI == 0), 4]
        # error_0 = 1.
        # # contourf(kI, kD, np.log10(error[:,:,4]))
        # # contourf(kI, kD, error[:,:,4] < error_0)
        # # contourf(kI, kD, np.log10(error[:,:,4]*(error[:,:,4] < error_0)))
        # contourf(kI, kD, np.log10((error[:,:,4]/error_0))*(error[:,:,4] <= error_0))
        # colorbar()
        # xlabel('kI')
        # ylabel('kD')

        # import sys
        # sys.exit(0)


    # imshow(error_trace)

    # For 1D parameter sweep
    if flag_experiment_1D_param_sweep:

        resolution = 100 # resolution of the controller gain study
        # I used 100 for the paper.


        gain_default = Gain(1., 0.0, 0., 0.05, 0.95)
        # gain_default = Gain(1., 0., 0.0, 0.3, 0.7)    

        # These ranges are for randomwalk, and correspond
        # to what was reported in the paper
        # param_name = 'kD'
        # param_range = (-0.2, 0.25) # Good range for kD (PE)
        # param_range = (-0.2, 0.45) # Good range for kD (control)
        
        # param_range = (0.2, 0.3) # Studying unstable region (PE)

        # param_name = 'kP'
        # param_range = (0.8, 1.25) # Good range for kP (PE)
        # param_range = (0.8, 1.25) # Good range for kP (control)    

        param_name = 'kI'
        param_range = (-0.5, 0.3) # Good range for kI (PE)
        # param_range = (-0.5, 1.2) # Good range for kI (control) 


        # These ranges are for two-state problem
        # param_name = 'kD'
        # param_range = (0., 1.0) # Good range for kD (PE)
        # param_range = (-0.2, 0.45) # Good range for kD (control)
        
        # param_name = 'kP'
        # param_range = (0.8, 1.25) # Good range for kP (PE)
        # param_range = (0.8, 1.25) # Good range for kP (control)    

        # param_name = 'kI'
        # param_range = (-1, 1) # Good range for kI (PE)
        # param_range = (-0.5, 1.2) # Good range for kI (control)    

        
        # These ranges are for the garnet problem
        # param_name = 'kD'
        # param_range = (0, 0.15) # Good range for kD (PE)
        # param_range = (-0.2, 0.45) # Good range for kD (control)
        
        # param_name = 'kP'
        # param_range = (0.8, 1.25) # Good range for kP (PE)
        # param_range = (0.8, 1.25) # Good range for kP (control)    

        # param_name = 'kI'
        # param_range = (0, 1) # Good range for kI (PE)
        # param_range = (-0.5, 1.2) # Good range for kI (control)    



        file_name_detail = ProblemType +'(discount='+discount_str+')-1D sweep-'+ param_name +\
                            ('(control)' if pi is None else '(PE)')
        
        experiment_1D_param_sweep(MDP, discount, pi, 
                                param_name=param_name,
                                param_range = param_range, resolution = resolution,
                                gain_default = gain_default,
                                iter_no = iter_no,
                                error_norm_ord = error_norm_ord, # Note that this is for the sup-norm
                                fig_filename=file_name_detail)

        if flag_experiment_1D_param_sweep_root_locus:
            file_name_detail = ProblemType +'(discount='+discount_str+')-root locus-'+ param_name +\
                                ('(control)' if pi is None else '(PE)')

            # Root locus doesn't make sense for the control case
            if pi is None:
                print("Warning! Root locus doesn't make sense for the control case.")
            
            # Set a dictionary of functions returning the error dynamics
            # matrix for each type of controllers
            P_controller = {'kP': lambda k: P_with_P(P, discount, k_p=k),
                            'kI': lambda ki: P_with_PI(P, discount, ki, beta = gain_default.I_beta, alpha=gain_default.I_alpha),
                            'kD': lambda kd: P_with_PD(P, discount, kd, D_term_type='basic')
                            }

            P = matrix(MDP.P[0])
            P_fn = P_controller[param_name]
            P_eig = plot_roots(P_fn, param_range = param_range,
                                fig_filename = file_name_detail)





    # For showing the performance for one specific case
    if flag_experiment_sample_behaviour:

        # gain_list = [Gain(1,0.3,0,0.05,0.95),
        #              Gain(0.7,0,0,0.05,0.95),
        #              Gain(1,0,0.2,0.05,0.95),
        #              Gain(1,-0.2,0.0,0.05,0.95),
        #              Gain(1,0.3,0.4,0.05,0.95)
        #              ]

        # Analytic solution for the PD controller -- only for reversible Markov chain
        # This doesn't really work for either Garnet or Chain Walk
        # (k_p_analyt, k_d_analyt) = kPD_analytic(discount)


        # Random Walk (PE)
        if pi:
            gain_list = [
                        Gain(1.2,0,0,0.05,0.95),
                        Gain(1,-0.4,0,0.05,0.95),
                        Gain(1,0,0.15,0.05,0.95),
                        # Gain(k_p_analyt, 0, k_d_analyt, 0.05, 0.95)
                        # Gain(1,-0.4,0.15,0.05,0.95),
                        ]

        # Random Walk (control)
        if pi is None:
            gain_list = [
                        Gain(1.2,0,0,0.05,0.95),
                        Gain(1,0.75,0,0.05,0.95),
                        Gain(1,0,0.4,0.05,0.95),
                        Gain(1,0.75,0.4,0.05,0.95),
                        Gain(1.,0.7,0.2,0.05,0.95),
                        # Gain(k_p_analyt, 0, k_d_analyt, 0.05, 0.95)
                        ]


        # Two-state (PE)
        # if pi:
        #     gain_list = [Gain(1.0,0,0.8,0.05,0.95),
        #                 Gain(1,0.8,0,0.05,0.95),
        #                 Gain(1,0,0.2,0.05,0.95),
        #                 Gain(k_p_analyt, 0, k_d_analyt, 0.05, 0.95)
        #                 #Gain(1,-0.4,0.15,0.05,0.95),
        #                 ]



        file_name_detail = ProblemType + '(discount='+discount_str+')-sample behaviour'+\
                            ('(control)' if pi is None else '(PE)')


        experiment_sample_behaviour(MDP, discount, pi, acc_param_list = gain_list,
                                    iter_no = iter_no,
                                    error_norm_ord = error_norm_ord,
                                    shown_states = 'adaptive', #[10,40],
                                    fig_filename = file_name_detail)


    # For showing the performance for the gain adaptation cases
    if flag_experiment_sample_behaviour_gain_adaptation:

        # Some sample values for the meta learning rate
        # These are selected for the randomwalk problem
        # if discount == 0.99:
        #     if pi:
        #         meta_lr = 0.05 #0.01 # (PE)
        #     else:
        #         meta_lr = 0.05 #0.005 # (control)
        # elif discount == 0.999:
        #     if pi:
        #         meta_lr = 0.005 # (PE)
        #     else:
        #         meta_lr = 0.001 # (control)

        # meta_lr = 0.002 # XXX
        # normalization_eps = 1e-12

        # These values work reasonably well for the Garnet problem.
        meta_lr = 0.05
        normalization_eps = 1e-20

        hyperparam_detail = '(eta,eps)=(' + str(meta_lr).replace('.','p') +\
                            ',' + str(normalization_eps).replace('.','p')+')'
    
        
        
        # Random Walk (PE)
        gain_list = [Gain(1.0,0,0,0.05,0.95),
                    # Gain(1.,0.5,0,0.05,0.95),
                    # Gain(0.9,0,0,0.05,0.95)
                    # Gain(1.,-0.5,0.0,0.05,0.95),
                    # Gain(1.,0.8,0,0.05,0.95)
                    # Gain(0.9,0.2,-0.2,0.05,0.95),
                    # Gain(1.,-0.2,0.2,0.05,0.95),
                    # Gain(1.5,0.0,0.0,0.05,0.95),
                    ]

        # gain_list = [Gain(1.2,0,0,0.05,0.95),
        #             Gain(1,-0.4,0,0.05,0.95),
        #             Gain(1,0,0.15,0.05,0.95)    
        #               #Gain(1,-0.4,0.15,0.05,0.95),
        #               ]

        # Random Walk (control)
        # gain_list = [Gain(1.2,0,0,0.05,0.95),
        #             Gain(1,0.75,0,0.05,0.95),
        #             Gain(1,0,0.4,0.05,0.95),
        #             Gain(1,0.75,0.4,0.05,0.95),
        #             Gain(1.,0.7,0.2,0.05,0.95)
        #              ]


        file_name_detail = ProblemType + '(discount='+discount_str+')-sample behaviour-adaptive gain'+\
                            hyperparam_detail + ('(control)' if pi is None else '(PE)')


        experiment_sample_behaviour_gain_adaptation(MDP, discount, pi, acc_param_list = gain_list,
                                    iter_no = iter_no,
                                    error_norm_ord = error_norm_ord,
                                    shown_states = 'adaptive', #[10,40],
                                    meta_lr = meta_lr, 
                                    normalization_eps = normalization_eps,
                                    normalization_flag = 'BE2',
                                    fig_filename = file_name_detail)


    if False:
        # Test!
        # This verifies whether the result of accelerated VI after k iterations is the same as 
        # predicted by the P matrix getting to the power of k.
        # This may become a unit test.
        P_pi_fn = lambda ki: P_with_PI(P, discount, ki, beta = gain_default.I_beta, alpha=gain_default.I_alpha)
        k_I = 0.0
        A = P_pi_fn(k_I)
        
        Vopt_true, Qopt_true, V_trace_orig_VI = \
                value_iteration(MDP.R, MDP.P, discount, IterationsNo=10*iter_no, policy=pi)
        
        e0 = vstack([0 - Vopt_true, np.zeros((state_size,1))])
        e_pred = (A**100)*e0
        
        Vopt_new, Qopt_new, V_trace_new, \
                Q_trace_new, dV_trace_new, dQ_trace_new, z_trace_new, \
                BE_Q_trace_new, BE_Q_integ_trace_new, gain_trace_new = \
                value_iteration_with_acceleration_new(MDP.R, MDP.P, discount, IterationsNo=iter_no,
                                                        alpha = 0., accelation_type='PID',
                                                        gain = Gain(1.,k_I,0.0,gain_default.I_alpha,gain_default.I_beta),
                                                        policy = pi)    
        
        e_actual = V_trace_new[:,99:100] - Vopt_true
        
        # figure()
        # k = 20
        # plot((A**(k+1))*e0,'k')
        # plot(V_trace_new[:,k:k+1] - Vopt_true,'r');
        
        # figure()
        # plot(sort(np.abs(eig(A)[0])))
        # plot(sort(np.abs(eig(P_pd_fn(0))[0])))