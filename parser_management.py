import argparse
import sys
import numpy as np


def get_args():

    parser = argparse.ArgumentParser(description='algorithms for Multi-Objective Optimization')

    parser.add_argument('--algs', type=str, help='algorithms', nargs='+', choices=['NWT', 'QNWT', 'MQNWT', 'LMQNWT', 'ArmijoBasedNSMA', 'WolfeBasedNSMA', 'LMQNWTBasedNSMA'])

    parser.add_argument('--probs', help='problems to evaluate', nargs='+', choices=['CEC', 'JOS', 'MFDS', 'MAN', 'MMR', 'MOP', 'MOP7', 'SLC'])

    parser.add_argument('--seeds', help='Seeds', type=int, nargs='+')

    parser.add_argument('--num_trials', help='Number of trials', type=int, nargs='+')

    parser.add_argument('--max_iter', help='Maximum number of iterations', default=None, type=int)

    parser.add_argument('--max_time', help='Maximum number of elapsed minutes per problem', default=None, type=float)

    parser.add_argument('--max_f_evals', help='Maximum number of function evaluations', default=None, type=int)

    parser.add_argument('--verbose', help='Verbose during the iterations', action='store_true', default=False)

    parser.add_argument('--verbose_interspace', help='Used interspace in the verbose (Requirements: verbose activated)', default=20, type=int)

    parser.add_argument('--plot_pareto_front', help='Plot Pareto front', action='store_true', default=False)

    parser.add_argument('--plot_pareto_solutions', help='Plot Pareto solutions (Requirements: plot_pareto_front activated; n in [2, 3])', action='store_true', default=False)

    parser.add_argument('--general_export', help='Export fronts (including plots), execution times and arguments files', action='store_true', default=False)

    parser.add_argument('--export_pareto_solutions', help='Export pareto solutions, including the plots if n in [2, 3] (Requirements: general_export activated)', action='store_true', default=False)

    parser.add_argument('--plot_dpi', help='DPI of the saved plots (Requirements: general_export activated)', default=100, type=int)

    ####################################################
    ### ONLY FOR Gurobi ###
    ####################################################

    parser.add_argument('--gurobi_method', help='Gurobi parameter -- Used method', default=-1, type=int)

    parser.add_argument('--gurobi_verbose', help='Gurobi parameter -- Verbose during the Gurobi iterations', action='store_true', default=False)

    ####################################################
    ### Armijo-Type Line Search ###
    ####################################################

    parser.add_argument('--ALS_alpha_0', help='ALS parameter -- Initial step size', default=1, type=float)

    parser.add_argument('--ALS_delta', help='ALS parameter -- Coefficient for the step size contraction', default=0.5, type=float)

    parser.add_argument('--ALS_beta', help='ALS parameter -- Coefficient for the sufficient decrease condition', default=1.0e-4, type=float)

    parser.add_argument('--ALS_min_alpha', help='ALS parameter -- Minimum possible value for the step size', default=1.0e-10, type=float)

    ####################################################
    ### ONLY FOR WolfeTypeLineSearch ###
    ####################################################

    parser.add_argument('--WLS_alpha_0', help='WLS parameter -- Initial step size', default=1, type=float)

    parser.add_argument('--WLS_beta', help='WLS parameter -- Beta', default=1.0e-4, type=float)

    parser.add_argument('--WLS_sigma', help='WLS parameter -- Sigma', default=1.0e-1, type=float)

    parser.add_argument('--WLS_tau', help='WLS parameter -- Tau', default=2.5, type=float)

    parser.add_argument('--WLS_gamma', help='WLS parameter -- Gamma (Theta)', default=0.5, type=float)

    parser.add_argument('--WLS_min_alpha', help='WLS parameter -- Min alpha', default=1.0e-10, type=float)

    parser.add_argument('--WLS_max_alpha', help='WLS parameter -- Max alpha', default=1.0e10, type=float)

    ####################################################
    ### ONLY FOR LMQNWT ###
    ####################################################

    parser.add_argument('--LMQNWT_max_cor', help='LMQNWT parameter -- Number of maximum memory for Modified L-BFGS', default=5, type=int)

    parser.add_argument('--LMQNWT_theta_for_stationarity', help='LMQNWT parameter -- Theta for Pareto stationarity', default=-5 * np.finfo(np.float).eps ** 0.5, type=float)

    ####################################################
    ### ONLY FOR NWT ###
    ####################################################

    parser.add_argument('--NWT_theta_for_stationarity', help='NWT parameter -- Theta for Pareto stationarity', default=-5 * np.finfo(np.float).eps ** 0.5, type=float)

    ####################################################
    ### ONLY FOR QNWT ###
    ####################################################

    parser.add_argument('--QNWT_theta_for_stationarity', help='QNWT parameter -- Theta for Pareto stationarity', default=-5 * np.finfo(np.float).eps ** 0.5, type=float)

    ####################################################
    ### ONLY FOR MQNWT ###
    ####################################################

    parser.add_argument('--MQNWT_theta_for_stationarity', help='MQNWT parameter -- Theta for Pareto stationarity', default=-5 * np.finfo(np.float).eps ** 0.5, type=float)

    ####################################################
    ### NSMA ###
    ####################################################

    parser.add_argument('--NSMA_pop_size', help='NSMA parameter -- Population size', default=100, type=int)

    parser.add_argument('--NSMA_crossover_probability', help='NSMA parameter -- Crossover probability', default=0.9, type=float)

    parser.add_argument('--NSMA_crossover_eta', help='NSMA parameter -- Crossover eta', default=20, type=float)

    parser.add_argument('--NSMA_mutation_eta', help='NSMA parameter -- Mutation eta', default=20, type=float)

    parser.add_argument('--NSMA_shift', help='NSMA parameter -- Shift parameter', default=10, type=float)

    parser.add_argument('--NSMA_crowding_quantile', help='NSMA parameter -- Crowding distance quantile', default=0.9, type=float)

    parser.add_argument('--NSMA_n_opt', help='NSMA parameter -- Number of iterations before doing optimization', default=5, type=int)

    parser.add_argument('--NSMA_FMOPG_max_iter', help='NSMA parameter -- Number of maximum iterations for FMOPG', default=10, type=int)

    parser.add_argument('--NSMA_theta_for_stationarity', help='NSMA parameter -- Theta for Pareto stationarity', default=-1.0e-10, type=float)

    parser.add_argument('--NSMA_theta_tol', help='NSMA parameter -- Theta tolerance', default=-1.0e-1, type=float)

    parser.add_argument('--NSMA_theta_dec_factor', help='NSMA parameter -- Theta decreasing factor', default=10 ** (-1 / 2), type=float)

    ####################################################
    ### LMQNWTBasedNSMA exclusives ###
    ####################################################

    parser.add_argument('--LMQNWTBasedNSMA_LMQNWT_max_iter', help='LMQNWTBasedNSMA parameter -- Number of maximum iterations for LMQNWT', default=10, type=int)

    parser.add_argument('--LMQNWTBasedNSMA_LMQNWT_max_cor', help='LMQNWTBasedNSMA parameter -- Number of maximum memory for LMQNWT', default=5, type=int)

    return parser.parse_args(sys.argv[1:])

