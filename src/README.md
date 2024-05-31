# RLAcceleration
Acceleration methods in RL


This is an incomplete reference guide.
The code doesn't accept arguments at the moment, and needs to be changed. The code itself, however, is relatively well-commented.

Two script files to run in order to get the empirical results are
* experiment_param_study.py
* experiment_gain_adaptation.py

The figures are generated when the code is run within the Spyder (Anaconda) with Python 3.7.6 and IPython 7.13.0. It also works if run by the command line.

The first one (experiment_param_study.py) is for most experiments, and the second one (experiment_gain_adaptation.py) is for the gain adaptation experiments with Garnet (which has randomness in the choice of domain).


## experiment_param_study.py

This is used to run many experiments reported in the paper.

There are a few flags that determine which experiment to run.
They are:

    flag_experiment_1D_param_sweep = False
    flag_experiment_2D_param_sweep = False
    flag_experiment_sample_behaviour = True
    flag_experiment_sample_behaviour_gain_adaptation = False
    flag_experiment_1D_param_sweep_root_locus = False

The problem type is determined by having only one of these lines uncommented:

    ProblemType = 'randomwalk'
    # ProblemType = 'garnet'

For Garnet, we have the parameters GarnetParam = (branching, non-zero reward),
which specifies the branching factor from each state-action pair, and the number of non-zero rewards, as the name suggests. For the experiments in the paper, we used (3,5).


The size of the problem is determined by:    
    state_size = 50
    action_size = 4
    
The discount factor is determined by:
    discount = 0.99
    
The variable pi determines what policy to compute.

    # Selecting the policy.
    # If None, it runs the VI with Bellman optimality (computing greedy policy at each step)
    # pi = [0]*state_size # Policy that always chooses the first action
    # pi = np.random.randint(0,2,state_size) # Random policy
    pi = None # For the Bellman optimality



For example, by having 

    flag_experiment_sample_behaviour = True
    ProblemType = 'randomwalk'
    pi = [0]*state_size
OR
    pi = None

we get Figure 1 (Chain Walk) Sample error behaviour for a 50-state chain walk problem for various accelerated variants of VI and the conventional VI.


As another example, by setting

    flag_experiment_sample_behaviour_gain_adaptation = True
    ProblemType = 'randomwalk'
    iter_no = 2000
    pi = None

we get Figure 4. (Chain Walk - Control) Gain adaptation results for (eta, eps) = (0.05, 10e-20).


To get the 1D sweeping experiments (e.g., Figures 2 and 3), we have to set
    flag_experiment_1D_param_sweep = True

Also we need to specify which gain should be varied and what range of gains should be computed. This is within 
    if flag_experiment_1D_param_sweep:
	...

and set by choosing param_name and param_range.

For example, 
    param_name = 'kD'
    param_range = (-0.2, 0.45)
    works fine for Chain Walk and Control problem.

There are some suggestions in the code.

The same for flag_experiment_2D_param_sweep. Note that this takes a longer time to compute. By changing 
    resolution = [100,100]
to a lower value, we can get a faster (and coarser) visualization.
	

## experiment_gain_adaptation.py

This is performing the gain adaptation experiment for the Garnet problem. The reason we have a different script is to handle the randomness in the choice of the Garnet problem and report the mean and std of performance, instead of just running a single experiment (which was suitable for Chain Walk).

The setting of the problem is as before:
    ProblemType = 'garnet'
    state_size = 50
    action_size = 4
    GarnetParam = (3,5)

The iteration number is set
    iter_no = 3000 # 3000 for 0.99, 10000 for 0.998

To decide between PE or Control, we have (as before):
```
# pi = [0]*state_size
# pi = np.random.randint(0,2,state_size)
pi = None
```

None refers to Control.


To decide whether the sweep is over eta or epsilon, we should uncomment either of the following parts of the code:

    # This is suitable if we want to sweep over epsilons

    # This is suitable if we want to sweep over eta

We get Figures 25-30 in the paper by this change.

In order to show the Best performance, the flag with_hp_model_selection = True when we call experiment_gain_adaptation_garnet(...) at the end of the script.


## ValueIteration.py
This implements the accelerated Value Iteration (P, PI, PID, and some other variants). 
The code has a lot of tracking and experimental sections.
The main functions there are
** value_iteration_with_acceleration_new
** adapt_gain_new

Out of these, only the first one is needed to be called outside this module.