"""
This code allows to obtain all the figures present in the paper.
"""

import numpy as np
from matplotlib.pyplot import *

from joblib import Parallel, delayed
import multiprocessing

from .vanilla_functions import *
from .finiteMDPs import *

from .Rootlocus_functions import *

rcParams['pdf.fonttype'] = 42
rcParams['ps.fonttype'] = 42
rcParams["figure.figsize"] = [9,8]

def eval_VI_perf(MDP, discount, policy = None, iter_no = 500, acceleration_type = None, acc_param_list = None,
                 reporting_iter = None, visualize = False, make_parallel = True,
                 error_norm_ord = np.inf,
                 ):

    # Only in case of a PID controller
    if acceleration_type is 'PID':
        print('PID controller') #, acc_param_list)
    else:
        return

    if reporting_iter is None:
        reporting_iter = iter_no-1

    # Initialize an empty list to save the errors
    error_trace = []

    # Compute the value function using the conventional VI
    Vopt_true, Qopt_true, V_trace_orig_VI = \
        value_iteration(MDP.R, MDP.P, discount, IterationsNo = 10 * iter_no, policy = policy)
    
    # If you want to speed up the computation: scan all the gains present in 
    if make_parallel:
        
        cpu_no = multiprocessing.cpu_count()
        print('Parallel running on ', cpu_no, ' CPUs!')
        VI_compact = lambda gain: value_iteration_with_acceleration_new(
                                    MDP.R, MDP.P, discount, IterationsNo=iter_no,
                                    alpha = 0., accelation_type='PID',
                                    gain = gain, policy = policy)[2]

        V_trace_acc_list = Parallel(n_jobs = cpu_no)(delayed(VI_compact)(gain) for gain in acc_param_list)

        
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
                semilogy(error_acc, label='VI with kI = '+str(gain.kI))  
                xlabel('Iteration')
                ylabel('||V_trace - Vopt_true||')
                legend()



    error_trace = np.array(error_trace)

    return error_trace

"""
This function allows to obatin the plots in Figure 2, as a function of which parameters are specified.
"""
def experiment_1D_param_sweep(MDP, discount, pi, param_name, param_range = (0, 1),
                                gain_default = None, resolution = 10, iter_no = 500,
                                reporting_iter = None, error_norm_ord = np.inf):

    # If no gain is specified by default
    if gain_default is None:
        print("Default gain")
        gain_default = Gain(1., 0., 0., 0.05, 0.95)

    # Create a dictionary of gains
    gain_dict = {'kP': lambda k: Gain(k, gain_default.kI, gain_default.kD, gain_default.I_alpha, gain_default.I_beta),
                 'kI': lambda k: Gain(gain_default.kP, k, gain_default.kD, gain_default.I_alpha, gain_default.I_beta),
                 'kD': lambda k: Gain(gain_default.kP, gain_default.kI, k, gain_default.I_alpha, gain_default.I_beta)
                }

    # Check if the wanted parameter is inside the dictionary
    if not(param_name in ['kD','kI', 'kP']):
        print("(experiment_1D_param_sweep) Incorrect name of the parameter. It should be one of kD, kI or kP")
        return

    # Create a range of values for the parameter   
    k_range = linspace(param_range[0], param_range[1], resolution)

    # Evaluate the lambda function associated to the specific parameter
    gain_fn = gain_dict[param_name]
    gain = [gain_fn(k) for k in k_range]

    # Here you decide how many lines should be displayed in the plots and their number
    if reporting_iter is None:
        reporting_iter = np.int_(np.array([1/4, 1/2, 3/4, 1.-1e-6]) * iter_no)
        print(reporting_iter)

    # Call the function defined above
    error_trace = eval_VI_perf(MDP, discount, policy = pi, iter_no = iter_no, acceleration_type = 'PID',
                acc_param_list = gain, reporting_iter = reporting_iter,
                make_parallel = True, error_norm_ord = error_norm_ord)

    figure(figsize = [7, 7])

    # Converting the norm to an appropriate label for the Y axis
    if error_norm_ord == np.inf:
        norm_label = '\infty'
    else:
        norm_label = str(error_norm_ord)

    # The label for the X axis: choose one among the strings present in the vector
    x_label = {'kD':'$k_D$', 'kP':'$k_P$', 'kI':'$k_I$'}[param_name]

    # plot(reporting_iter,error_trace.transpose())
    semilogy(k_range, error_trace, linewidth = 2)
    xlabel(x_label, fontsize = 35)

    if pi is None:
        ylabel('$\||V_k - V^*\||_{'+norm_label+'}$', fontsize = 25)
        title('Exp 3 - Fig 2 - Control case', fontsize = 20)
    else:
        ylabel('$\||V_k - V^{\pi}\||_{'+norm_label+'}$', fontsize = 25)
        title('Exp 3 - Fig 2 - Prediction case', fontsize = 20)

    # ylabel('$\||V - V_k\||_{'+norm_label+'}$', fontsize = 20)
    legend(reporting_iter, fontsize=20)
    # semilogy(k_range, error_trace,'kx', markeredgewidth = 0.5)
    semilogy(k_range[::3], error_trace[::3],'kx', markeredgewidth = 0.5)
    axis("tight")
    grid(True,which='both')
    tick_params(labelsize=15)


"""
This method allows to obtain the plots 1a) and 1b) in the paper, and also some results presented in the Appendix. Basically, it allows to see 
how the error varies as the number of iterations increases, or by varying the number of states.
"""
def experiment_sample_behaviour(MDP, discount, pi, acc_param_list = None,
                                iter_no = 500,
                                error_norm_ord = np.inf,
                                visualize_state_error = True,
                                shown_states = None,
                                gain_list = None):

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

    """
    The first figure allows to see Figure 1a) or 1b) in the paper, that is how the error varies as the number of iterations increases.
    """
    figure(figsize = [7, 7])
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
        title('Exp 1 - Figure 1 - Control case', fontsize = 20)
    else:
        ylabel('$\||V_k - V^{\pi}\||_{'+norm_label+'}$', fontsize = 25)
        title('Exp 1 - Figure 1 - Prediction case', fontsize = 20)
    # legend(['VI (original)','VI with acceleration'])
    legend(fontsize = 18, framealpha=0.5)
    # axis("tight")
    grid(True,which='both')
    xticks(fontsize = 15)
    yticks(fontsize = 15)

    # Setting the number of rows and columns of the subplot
    # The rows can be of the larger size.
    col_no = int(np.floor(np.sqrt( len(gain_list)+1 )))
    row_no = int(np.ceil(np.sqrt( len(gain_list)+1 )))

    if pi is None:
        y_label = '$V_k(x) - V^*(x)$'
    else:
        y_label = '$V_k(x) - V^{\pi}(x)$'


    state_no = Vopt_true.shape[0]
    if shown_states is None:
        state_ind = arange(0,state_no)
    elif shown_states == 'adaptive':
        #print(f"The error may be {max(3, 1)}")
        state_ind = np.arange(0,state_no, np.max([int(state_no/5), 1]) )
        # state_ind = np.arange(0,state_no, 1 )
    else:
        state_ind = shown_states
    # Otherwise, it is assumed to be a list of states

    # print('state_ind',state_ind)
    
    # fig, ax = subplots(row_no, col_no)

    # figure()
    # plot(Vopt_true)

    """
    the second figure allows to plot the Bellman error (PE = Vk - V_pi, for the prediction case; or Vk-V* for control) by varying the number
    of states considered in the problem (0, 10, 20, 30, 40)
    """
    figure(figsize = [7, 7])   
    subplot(row_no, col_no,1)
    subplots_adjust(wspace = 0.3, hspace = 0.5)
    # plot(V_trace_conv_VI.transpose() - Vopt_true[:,0])    
    plot(V_trace_conv_VI[state_ind,:].transpose() - Vopt_true[state_ind].squeeze(), linewidth = 1)
    title('Exp 1 - VI (Conventional)', fontsize=10)
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

        title('Exp 1 - $BE_2$ - VI(PID) with $(k_p, k_I, k_d)=$('+\
                str(gain.kP)+', '+str(gain.kI)+', '+str(gain.kD)+')', fontsize=10)

        xlabel('Iteration', fontsize=10)
        ylabel(y_label, fontsize=10)
        
        # Only show the legend if the number of states in state_ind isn't too large.
        # The choice of 5 is arbitrary here, and possibly can become a parameter.
        if len(state_ind) <= 5:
            legend(labels=state_ind, loc = 'upper right', fontsize = 5)

        grid(True,which='both')

"""
This method allows to obtain the plots 3a) and 3b) in the paper, and also some results presented in the Appendix 
(error variation by varying the number of states).
"""
def experiment_sample_behaviour_gain_adaptation(MDP, discount, pi, acc_param_list = None,
                                iter_no = 500,
                                error_norm_ord = np.inf,
                                visualize_state_error = True,
                                shown_states = None,
                                meta_lr = None, 
                                normalization_eps = None, normalization_flag = 'BE2',
                                gain_list = None):

    # print(acc_param_list)

    # Converting the norm to an appropriate for the Y axis label
    if error_norm_ord == np.inf:
        norm_label = '\infty'
    else:
        norm_label = str(error_norm_ord)

    print ('Computing the value functions ...')
    
    Vopt_true, Qopt_true, V_trace_orig_VI = \
        value_iteration(MDP.R, MDP.P, discount, IterationsNo=10*iter_no, policy = pi)
        
    # Call the value iteration method for the proportional controller
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

    # Call the value iteration method for the PID controller (and plot Figure 3b) by setting True to gain_adaptation)
    for gain in acc_param_list:
        Vopt_new, Qopt_new, V_trace_new, \
            Q_trace_new, dV_trace_new, dQ_trace_new, z_trace_new, \
            BE_Q_trace_new, BE_Q_integ_trace_new, gain_trace_new = \
            value_iteration_with_acceleration_new(MDP.R, MDP.P, discount, IterationsNo=iter_no,
                                                    alpha = 0., accelation_type='PID',
                                                    gain = gain,
                                                    gain_adaptation = True,
                                                    meta_lr = meta_lr, 
                                                    normalization_eps = normalization_eps,
                                                    normalization_flag = normalization_flag,
                                                    policy = pi)

        error_acc.append(np.linalg.norm(V_trace_new - Vopt_true, axis = 0, ord=error_norm_ord))
        V_trace_acc.append(V_trace_new)
        gain_trace_acc.append(gain_trace_new)


    line_style_list = ['-', '--', '-.', ':']
    
    """
    Figure 3a
    """
    figure(figsize = [7, 7])
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
        title('Exp 2 - Fig 3 - Control case', fontsize = 20)
    else:
        ylabel('$\||V_k - V^{\pi}\||_{'+norm_label+'}$', fontsize = 25)
        title('Exp 2 - Fig 3 - Prediction case', fontsize = 20)
    # legend(['VI (original)','VI with acceleration'])
    legend(fontsize = 15, loc='upper right')
    # axis("tight")
    grid(True,which='both')
    xticks(fontsize = 15)
    yticks(fontsize = 15)
    
    """
    Better visualization of the gains (Figure 3b)
    """
    figure(figsize = [7, 7])
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
    xlabel('Iteration', fontsize = 15)
    ylabel('Gains $k_p, k_d, k_I$', fontsize = 15)

    # # ylabel('||V_trace - Vopt_true||', fontsize = 20)
    # if pi is None:
    #     ylabel('$\||V_k - V^*\||_{'+norm_label+'}$', fontsize = 20)
    # else:
    #     ylabel('$\||V_k - V^{\pi}\||_{'+norm_label+'}$', fontsize = 20)
    # # legend(['VI (original)','VI with acceleration'])
    legend(['$k_p$', '$k_I$', '$k_d$'], fontsize = 20)
    title('Exp 2 - Figure 3b - Controller gains', fontsize = 20)
    # axis("tight")
    grid(True,which='both')
    xticks(fontsize = 15)
    yticks(fontsize = 15)

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
        state_ind = np.arange(0, state_no, max([int(state_no/5), 1]) )
    else:
        state_ind = shown_states
    # Otherwise, it is assumed to be a list of states

    # print('state_ind',state_ind)
    
    # fig, ax = subplots(row_no, col_no)

    # figure()
    # plot(Vopt_true)

    """ 
    Plot the error variation as a functiuon of the iterations, by varying the number of states
    """    
    figure()   
    subplot(row_no, col_no,1)
    subplots_adjust(wspace = 0.3, hspace = 0.5)
    # plot(V_trace_conv_VI.transpose() - Vopt_true[:,0])    
    plot(V_trace_conv_VI[state_ind,:].transpose() - Vopt_true[state_ind].squeeze(), linewidth = 1)
    title('Exp 2 - VI (Conventional)', fontsize = 20)
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
                str(gain.kP)+', '+str(gain.kI)+', '+str(gain.kD)+')', fontsize = 20)

        xlabel('Iteration', fontsize=10)
        ylabel(y_label, fontsize=10)
        
        # Only show the legend if the number of states in state_ind isn't too large.
        # The choice of 5 is arbitrary here, and possibly can become a parameter.
        if len(state_ind) <= 5:
            legend(labels=state_ind, loc = 'upper right', fontsize = 5)

        grid(True,which='both')

        
def measure_error_perf(error):
    return np.mean(np.log(error+1e-20), axis = 1)

# This function is used in the second function to mainly return the errors happened during the gain adaptation (change in the hyperparameters)
def evaluate_gain_adaptation(MDP, discount, pi, hyper_param_list,
                            init_gain = Gain(1, 0, 0, 0.05, 0.95),
                            iter_no = 500,
                            error_norm_ord = np.inf,
                            gain_adaptation = True, 
                            normalization_flag = 'BE2',
                            make_parallel = True,
                            gain = None):

    # Call the function to obtain the true value function (used as a reference in the error calculation)
    Vopt_true, _, _ = \
        value_iteration(MDP.R, MDP.P, discount, IterationsNo = 10 * iter_no, policy = pi)
    
    # Define some empty lists  
    error_list = []
    V_trace_list = []
    gain_trace_list = []

    if make_parallel: # In case you want to parallelize evaluations
        
        cpu_no = multiprocessing.cpu_count()
        print('Parallel running on ', cpu_no, ' CPUs!')

        # Import the function defined in 'ValueIteration.py': VI_compact is a list of all the values returned by 'value_iteration_with_acceleration_new'
        # NOTE: return V, Q, V_trace, Q_trace, dV_trace, dQ_trace, z_trace, BE_Q_trace, BE_Q_integ_trace, gain_trace
        VI_compact = lambda meta_lr, normalization_eps: value_iteration_with_acceleration_new(
                                            MDP.R, MDP.P, discount, IterationsNo=iter_no,
                                            alpha = 0., accelation_type='PID',
                                            gain = init_gain,
                                            gain_adaptation = True, # You want to find the best gains following the gain adaptation procedure
                                            meta_lr = meta_lr,
                                            normalization_flag = normalization_flag,
                                            normalization_eps = normalization_eps,
                                            policy = pi)

        """
        Run the function in parallel, varying the hyperparameters eta and epsilon.
        The block (delayed(VI_compact)(meta_lr, normalization_eps) effectively means 'call in parallel VI_compact with the arguments meta_lr, normalization_eps'
        """
        trace_list = Parallel(n_jobs=cpu_no)(delayed(VI_compact)(meta_lr, normalization_eps ) for 
                                        meta_lr, normalization_eps in hyper_param_list)

        # Loop through all the elements in the list (each element is driven by a different hyperparameter pair)
        for output_trace in trace_list:
            V_trace_new = output_trace[2] # V_trace, third element of the list
            gain_trace_new = output_trace[-1] # gain_trace

            # Calculate the error 
            error_list.append(np.linalg.norm(V_trace_new - Vopt_true, axis = 0, ord = error_norm_ord))
            V_trace_list.append(V_trace_new)
            gain_trace_list.append(gain_trace_new)

    else: # In case you do not want to parallelize evaluations
        for (meta_lr, normalization_eps) in hyper_param_list: # This is the big disadvantage of this code: it is not parallelized, so you need a for loop
            print(meta_lr, normalization_eps)

            if meta_lr is None:
                meta_lr, normalization_eps = 0, 0

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

    # return the error list, the history of the value functions, the history of the gains, and the optimal value function 
    return error_list, V_trace_list, gain_trace_list, Vopt_true

# Core function
def experiment_gain_adaptation_garnet(discount, pi, hyper_param_list,
                                state_size = 100, action_size = 4,
                                GarnetParam = (3,5),
                                runs_no = 2,
                                init_gain = Gain(1, 0, 0, 0.05, 0.95),
                                with_hp_model_selection = False,
                                normalization_flag = 'BE2',
                                iter_no = 500,
                                error_norm_ord = np.inf):

    # Define some empty lists
    normalized_conv_VI_error = []
    normalized_acc_error = []
    normalized_best_error = []

    # Usually, you run this par 20 times
    for run in range(runs_no):

        print('Run:', run)
        
        # Generate a new MDP (no choice: you want to focus on Garnet)
        MDP = FiniteMDP(StateSize = state_size, ActionSize = action_size,
                                    ProblemType='garnet', GarnetParam = GarnetParam)
        
        # Call the function defined above to evaluate the gain adaptation
        (error_list, V_trace_list, gain_trace_list, Vopt_true) = \
        evaluate_gain_adaptation(MDP, discount, pi, hyper_param_list = hyper_param_list + [(0, 0)],
                        iter_no = iter_no, error_norm_ord = error_norm_ord, 
                        gain_adaptation = True, normalization_flag = normalization_flag)
   
        # The last item is for the conventional VI
        error_conv_VI = error_list[-1]

        # Normalize the errors
        error_list = np.array(error_list[:-1])
        Vopt_true_norm = np.linalg.norm(Vopt_true,ord = error_norm_ord)
        normalized_acc_error.append(error_list / Vopt_true_norm)
        normalized_conv_VI_error.append( error_conv_VI / Vopt_true_norm)

        # Find the best hyperparameter
        if with_hp_model_selection:

            # Call the function defiend above: np.mean(np.log(error_list + 1e-20), axis = 1) ==> this is the performance index we choose
            perf_acc = measure_error_perf(error_list)
            print('perf_acc:', perf_acc)
            hp_best = np.argmin(perf_acc) # Find the index corresponding to the best hyperparameter
            error_best = error_list[hp_best]
            print('New!:', hp_best, hyper_param_list[hp_best])
            
            # Calculate the normalized error
            normalized_best_error.append(error_best / Vopt_true_norm)

        # Display the normalized normalized true value function 
        print('||Vopt_true||=',Vopt_true_norm )

    # Augment the lisst associated to all the possible runs
    normalized_conv_VI_error = np.array(normalized_conv_VI_error)
    normalized_acc_error = np.array(normalized_acc_error)
    normalized_best_error = np.array(normalized_best_error)

    # Visualization part
    if error_norm_ord == np.inf:
        norm_label = '\infty'
    else:
        norm_label = str(error_norm_ord)

    figure()
    
    iteration_range = np.arange(normalized_conv_VI_error.shape[-1]) # x axis

    # Display both mean and variance of the error associate to teh conventional VI (of all the runs, you obbtain a single 'mean' sequence)
    err_mean = np.mean(normalized_conv_VI_error, axis = 0)
    err_stderr = np.std(normalized_conv_VI_error, axis = 0) / np.sqrt(runs_no)
    semilogy(err_mean,label='Conventional VI',linewidth = 2)
    fill_between(x = iteration_range, y1=err_mean - err_stderr, y2=err_mean + err_stderr, alpha = 0.1) # color in between the luines defining the standard eviation

    print('err_stderr:', np.mean(err_stderr))

    line_style_list = ['-', '--', '-.']
    
    # Loop through all the hyperparameters to display all teh mean errors associated to the accelerated variants
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

    # If you also weant to display additional information about the sequence with the best performances
    if with_hp_model_selection:
        err_mean = np.mean(normalized_best_error, axis = 0)
        err_stderr = np.std(normalized_best_error, axis = 0) / np.sqrt(runs_no)
        semilogy(err_mean,label='Best',linewidth = 2,color='r')
        fill_between(x = iteration_range, y1=err_mean - err_stderr, y2=err_mean + err_stderr,
                    color='r', alpha = 0.1)

    xlabel('Iteration', fontsize = 15)
    
    if pi is None: # Control
        ylabel('$\||V_k - V^*\||_{'+norm_label+'} / \||V^*\||_{'+norm_label+'}$', fontsize = 25)
        title('No policy $\pi$ - Control', fontsize = 20)
    else: # Prediction
        ylabel('$\||V_k - V^{\pi}\||_{'+norm_label+'} / \||V^{\pi} \||_{'+norm_label+'}$', fontsize = 25)
        title('Prediction', fontsize = 20)
    legend(fontsize = 12, framealpha=0.5) # 15 for most figures
    axis("tight")
    grid(True,which='both')
    xticks(fontsize = 15)
    yticks(fontsize = 15)
    #show()


