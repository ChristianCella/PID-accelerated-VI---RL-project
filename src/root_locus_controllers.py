from numpy import *
from numpy.linalg import eig
from numpy.linalg import inv

from matplotlib.pyplot import *

import numpy as np

def plot_roots(P_func = None, param_range = (0,1), fig_filename = None):


    if P_func is None:
        return
    
    param_values = linspace(param_range[0],param_range[1],200) #200
    
    param_value_labelled = [int(len(param_values)/4),int(len(param_values)/2),
                            int(len(param_values)*3/4)]
    # par_len = len(param_values)
    # print(par_len/2)
    
    P_dim = P_func(0).shape[0]
    P_eig = array([eig(P_func(theta))[0] for theta in param_values])

    # fig, (ax1,ax2) = subplots(2,1) #, sharey=True)
    fig, (ax1) = subplots(1,1) #, sharey=True)
    
    # print(np.max(P_eig.real), np.min(P_eig.real))
    x_max = max(1.1, np.max(P_eig.real) + 0.1 )
    x_min = min(-1.1, np.min(P_eig.real) - 0.1 )

    y_max = max(1.1, np.max(P_eig.imag) + 0.1 )
    y_min = min(-1.1, np.min(P_eig.imag) - 0.1)

    
    ax1.set_xlim(x_min,x_max)
    ax1.set_ylim(y_min,y_max)
    ax1.add_artist( Circle( (0,0),1., fill=False, linewidth=2) )
    # ax1.add_artist( Circle( (0,0),0.5, linestyle = '--', fill=False) )

    grid("on")
    axis("on")
    xlabel('Real', fontsize = 20)
    ylabel('Imaginary', fontsize=20)
    
    radius_range = [0.25, 0.5, 0.75]
    [ax1.add_artist( Circle( (0,0),rad, linestyle = ':', fill=False) ) \
     for rad in radius_range]

    for eig_no in range(P_dim):
        # print(eig_no)
        
        ax1.plot(P_eig[0,eig_no].real, P_eig[0,eig_no].imag,'rx',
                markeredgewidth=2, markersize=15)

        ax1.plot(P_eig[-1,eig_no].real, P_eig[-1,eig_no].imag,'ro',
                markeredgewidth=2, fillstyle = 'none', markersize=15)
        ax1.plot(P_eig[:,eig_no].real, P_eig[:,eig_no].imag,'b.', markersize=1)
        
        # I don't think this is a good idea. The returned eigenvalues do not 
        # necessarily correspond to the same one.
        # ax1.plot(P_eig[0:int(par_len/2),eig_no].real, P_eig[0:int(par_len/2),eig_no].imag,'m.', markersize=2)

        # ax1.text(P_eig[0,eig_no].real, P_eig[0,eig_no].imag + 0.05, 'Here!', fontsize='x-large')

        for label_ind in param_value_labelled:
            # print(P_eig[label_ind,eig_no].imag/(P_eig[label_ind,eig_no].real+1e-6))
            arrow_theta = np.arctan(P_eig[label_ind,eig_no].imag/(P_eig[label_ind,eig_no].real+1e-6))
            arrow_theta += pi/2
            arrow_dx = 0.2*cos(arrow_theta)
            arrow_dy = 0.2*sin(arrow_theta)
            arrow_xy = (P_eig[label_ind,eig_no].real,P_eig[label_ind,eig_no].imag)
            arrow_text_xy = (P_eig[label_ind,eig_no].real + arrow_dx, P_eig[label_ind,eig_no].imag + arrow_dy)
            annot_text = str('{:.2f}'.format(param_values[label_ind]))

            # ax1.annotate(annot_text,xy=arrow_xy,
            #             arrowprops=dict(arrowstyle='fancy', facecolor='black') )

            ax1.plot(P_eig[label_ind,eig_no].real,P_eig[label_ind,eig_no].imag,'mo', markersize = 3)
            # ax1.annotate(annot_text,xy=arrow_xy, xytext = arrow_text_xy, fontsize = 10,
            #             arrowprops=dict(arrowstyle='->', facecolor='black') )

        # ax1.text(P_eig[0,eig_no].real, P_eig[0,eig_no].imag*1.1 + 0.05, str(eig_no), fontsize='x-large')

    # if fig_filename is not None:
    #     ax1.savefig('figures/'+fig_filename+'.pdf')
    

    fig2 = figure()
    dom_eig = np.max(abs(P_eig), axis = 1)
    plot(param_values, dom_eig,'.')
    print('min |lambda_max(P)| is achieved at ', param_values[argmin(dom_eig)])

    
    # l2-induced norm
    # XXX Maybe make it a flag
    if False:
        P_norm = array([np.linalg.norm(P_func(theta), ord=2) for theta in param_values])
        plot(param_values, P_norm, 'k.')
        print('min ||P||_2 is achieved at ', param_values[argmin(P_norm)])

    # ax2.plot(param_values, np.max(abs(P_eig), axis = 1),'.')
    grid("on")
    axis("on")
    xlabel('Controller gain', fontsize = 20)
    ylabel('Dominant eigenvalue', fontsize=20)

    if fig_filename is not None:
        fig.savefig('figures/'+fig_filename+'.pdf')
        fig2.savefig('figures/'+fig_filename+'(modulus).pdf')


    
    
    # ax2.plot(param_values, np.abs(P_eig))
#    ax2.set
#    ax2.xlabel('Parameter')
#    ax2.ylabel('Largerst eigenvalue')

        
        
    return P_eig


def P_with_PD(P, discount, k_d, D_term_type = 'basic'):
    """Compute the error dynamics matrix for the PD controller.
    
    V_{k+1} = Tpi V_k + k_d D (V_k - V_{k-1})
    D_term_type:
    - If 'basic', D is the identity matrix (leading to a scalar gain).
    - If 'P', D is P.

    Use 'basic' as a default.
    """

    state_size = P.shape[0]
    if D_term_type == 'basic':
        D_matrix = eye(state_size)
    elif D_term_type == 'P':
        D_matrix = P

    A = vstack([        
            hstack([discount*P + k_d*D_matrix, -k_d*D_matrix]),
            hstack([eye(state_size), 0*eye(state_size)])
            ]
        )
    
    return A


def P_with_PD_general(P, discount, k_p, k_d, D_term_type = 'basic'):
    """Compute the error dynamics matrix for the PD controller.
    
    V_{k+1} = Tpi V_k + k_d D (V_k - V_{k-1})
    D_term_type:
    - If 'basic', D is the identity matrix (leading to a scalar gain).
    - If 'P', D is P.

    Use 'basic' as a default.
    """

    state_size = P.shape[0]
    if D_term_type == 'basic':
        D_matrix = eye(state_size)
    elif D_term_type == 'P':
        D_matrix = P

    A = vstack([        
            hstack([(1 - k_p)*eye(state_size) + k_p*discount*P + k_d*D_matrix, -k_d*D_matrix]),
            hstack([eye(state_size), 0*eye(state_size)])
            ]
        )
    
    return A


def P_with_PI(P, discount, k_i, beta = 0.9, alpha = None): #, D_term_type = 'basic'):
    """Compute the error dynamics matrix for the PI controller.
    
    alpha and beta are the same as in the paper.
    z_{k+1} = beta z_k + alpha BR(V_k)
    V_{k+1} = Tpi V_k + k_i z_{k+1}

    If alpha isn't given, it is set to alpha = 1 - beta.
    """

    if alpha is None:
        alpha = 1 - beta

    state_size = P.shape[0]
    # if D_term_type == 'basic':
    #     D_matrix = eye(state_size)
    # elif D_term_type == 'P':
    #     D_matrix = P

    A = vstack([        
            hstack([discount*(1 + k_i*alpha)*P - k_i*alpha*eye(state_size), k_i*beta*eye(state_size)]),
            hstack([alpha*(discount*P - eye(state_size)), beta*eye(state_size)])
            ]
        )

    # I think this is wrong. Should be removed. XXX
#     A = vstack([        
#             hstack([discount*P, k_i*eye(state_size)]),
#             hstack([0.1*(discount*P - eye(state_size)), 0.9*eye(state_size)])
# #            hstack([(0.1*eye(state_size)), 0.8*eye(state_size)])            
#             ]
#         )
    
    return A


def P_with_PID(P, discount, k_p = 1., k_i = 0., k_d = 0., beta = 0.9, alpha = None): #, D_term_type = 'basic'):
    """Compute the error dynamics matrix for the PID controller.
    
    alpha and beta are the same as in the paper. XXX
    z_{k+1} = beta z_k + alpha BR(V_k) XXX
    V_{k+1} = Tpi V_k + k_i z_{k+1} XXX

    If alpha isn't given, it is set to alpha = 1 - beta. XXX
    """

    if alpha is None:
        alpha = 1 - beta

    state_size = P.shape[0]
    # if D_term_type == 'basic':
    #     D_matrix = eye(state_size)
    # elif D_term_type == 'P':
    #     D_matrix = P

    # I write it separately as it is long
    term_11 = (1. - k_p)*eye(state_size) + discount*k_p*P +\
                 alpha*k_i*(discount*P - eye(state_size)) + k_d*eye(state_size)

    A = vstack([        
            hstack([term_11, -k_d*eye(state_size), beta*k_i*eye(state_size)]),
            hstack([eye(state_size), 0*eye(state_size), 0*eye(state_size)]),
            hstack([alpha*(discount*P - eye(state_size)), 0*eye(state_size), beta*eye(state_size)])
            ]
            )
    
    return A

def P_with_P(P, discount, k_p):
    """Compute the error dynamics matrix for the P controller.
    
    V_{k+1} = (1 - k_p) V_k + k_p Tpi V_k
    """

    state_size = P.shape[0]

    A = (1 - k_p)*(np.eye(state_size)) + k_p*discount*P

    return A



if __name__ == '__main__':

    import FiniteMDP
    state_size = 50
    # MDP = FiniteMDP.FiniteMDP(state_size, ProblemType='randomwalk') # ,GarnetParam=(3,2))
    
    # MDP = FiniteMDP.FiniteMDP(2,ActionSize = 1, ProblemType='TwoThreeStatesProblem',TwoThreeStatesParam = 0.95)
    MDP = FiniteMDP.FiniteMDP(3,ActionSize = 1, ProblemType='TwoThreeStatesProblem', TwoThreeStatesParam = 1/3)

    # P = matrix(MDP.P[0])
    
    eps = 1e-8
    P = matrix([[1-eps, eps], [eps, 1-eps]])
    eig(P)[0]

    # state_size = 2
#    state_size = 10 #P.shape[0]
##    P = [[0.1,1],[0.03, 0.97] ]
    # P = [[1., 0.2,2], [1, 1., 0], [1., 0., 1]]
#   #P = [[1.0]]
    # P = eye(state_size)
    # P = random.random( (state_size,state_size) )    
#   # P = P**2
##    P = eye(2)

    # Simple 2 state problem
    # P = [[0.7, 0.3],
    #       [0.3, 0.7]]

    # p = 0.4
    # P = [[1-p, p],
    #       [p, 1-p]]


    # P = [[0.0, 1.0],
    #       [0.5, 1.0]]

    
    # Simple 3 state problem with complex eigenvalues
    # P = [[0.0, 0.0, 1.],
    #       [1., 0.0, 0.],
    #       [0.5, 0.5, 0.0]]

    # P = [[0.0, 1., 0],
    #       [0., 0.0, 1],
    #       [1., 0, 0.0]]


    # p = 2./9
    # P = [[1-3*p, 2*p, p],
    #       [p, 1-3*p, 2*p],
    #       [2*p, p, 1-3*p]]

    # p = 1/4
    # P = [[1/3, 1/3+p, 1/3-p],
    #       [1/3-p, 1/3, 1/3+p],
    #       [1/3+p, 1/3-p, 1/3]]


    # P = [[0, 1, 0],
    #     [0, 0, 1],
    #     [1, 0, 0]]

    # P = [[0, 1, 0],
    #     [0, 0, 1],
    #     [1, 0, 0]]

    # P = [[0.1, 0.9, 0],
    #     [0, 0.1, 0.9],
    #     [0.9, 0, 0.1]]

    P = matrix(P)
    P = P/sum(P,axis = 1)
    
    print(eig(P))
    
    discount = 0.99

    P_fn = lambda p: (1-p)*eye(state_size) + p*P

    P_pd_basic_fn = lambda kd: P_with_PD(P, discount, kd, D_term_type='basic')
    P_pd_P_fn = lambda kd: P_with_PD(P, discount, kd, D_term_type='P')
    P_pi_fn = lambda ki: P_with_PI(P, discount, ki, beta = 0.95)
    P_pid_fn = lambda ki: P_with_PID(P, discount, k_p=1.1, k_i = ki, k_d = 0.0, beta = 0.3)
    P_p_fn = lambda kp: P_with_P(P, discount, kp)
    
    
    from eigen_study_PID_VI import kPD_analytic
    
    (kp_star, kd_star) = kPD_analytic(discount)
    P_pd_gen_fn = lambda kd: P_with_PD_general(P, discount, k_p=kp_star, k_d = kd)
    
    
#    print(P_fn(0))
#    P_eig = array([eig(P_fn(p))[0] for p in linspace(0,1)])    
    
    # plot_roots(P_fn)
#    P_eig = plot_roots(P_pd_basic_fn, param_range = (0.0,0.3))
    # P_eig = plot_roots(P_pd_P_fn, param_range = (0.0,1.))

    # P_eig = plot_roots(P_pi_fn, param_range = (0,10),
    #                     fig_filename = None) # = 'roots_PI')    

    # P_eig = plot_roots(P_pd_basic_fn, param_range = (0,1),
    #                     fig_filename = None) # = #'roots_PD')

    # P_eig = plot_roots(P_pd_gen_fn, param_range = (0,2),
    #                     fig_filename = None) # = #'roots_PD')


    P_eig = plot_roots(P_pid_fn, param_range = (0,50),
                        fig_filename = None) # = 'roots_PID')    

    # P_eig = plot_roots(P_p_fn, param_range = (0.8,1.2),
    #                     fig_filename = None) # = 'roots_PID')    

