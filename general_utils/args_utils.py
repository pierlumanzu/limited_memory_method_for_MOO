import os

from constants import PROBLEMS, PROBLEM_DIMENSIONS


def print_parameters(args):
    if args.verbose:
        print()
        print('Parameters')
        print()

        for key in args.__dict__.keys():
            print(key.ljust(args.verbose_interspace), args.__dict__[key])
        print()


def check_args(args):

    for s in args.seeds:
        assert s > 0

    for nt in args.num_trials:
        assert nt > 0

    if args.max_iter is not None:
        assert args.max_iter > 0
    if args.max_time is not None:
        assert args.max_time > 0
    if args.max_f_evals is not None:
        assert args.max_f_evals > 0

    assert args.verbose_interspace >= 1
    assert args.plot_dpi >= 1

    assert -1 <= args.gurobi_method <= 5

    assert args.ALS_alpha_0 > 0
    assert 0 < args.ALS_delta < 1
    assert 0 < args.ALS_beta < 1
    assert args.ALS_min_alpha > 0

    assert args.WLS_alpha_0 > 0
    assert 0 < args.WLS_beta < 1
    assert 0 < args.WLS_sigma < 1
    assert args.WLS_beta < args.WLS_sigma
    assert args.WLS_tau > 1
    assert 0.5 <= args.WLS_gamma < 1
    assert args.WLS_min_alpha > 0
    assert args.WLS_max_alpha > 0
    assert args.WLS_min_alpha < args.WLS_max_alpha

    assert args.NWT_theta_for_stationarity <= 0

    assert args.QNWT_theta_for_stationarity <= 0

    assert args.MQNWT_theta_for_stationarity <= 0

    assert args.LMQNWT_max_cor > 0
    assert args.LMQNWT_theta_for_stationarity <= 0

    assert args.NSMA_pop_size > 0
    assert 0 <= args.NSMA_crossover_probability <= 1
    assert args.NSMA_crossover_eta >= 0
    assert args.NSMA_mutation_eta >= 0
    assert args.NSMA_shift > 0
    assert 0 <= args.NSMA_crowding_quantile <= 1
    assert args.NSMA_n_opt > 0
    assert args.NSMA_FMOPG_max_iter > 0
    assert args.NSMA_theta_for_stationarity <= 0
    assert args.NSMA_theta_tol <= 0
    assert args.NSMA_theta_tol < args.NSMA_theta_for_stationarity
    assert 0 < args.NSMA_theta_dec_factor < 1

    assert args.LMQNWTBasedNSMA_LMQNWT_max_iter > 0
    assert args.LMQNWTBasedNSMA_LMQNWT_max_cor > 0


def args_preprocessing(args):
    check_args(args)

    algorithms_names = []

    if 'NWT' in args.algs:
        algorithms_names.append('NWT')

    if 'QNWT' in args.algs:
        algorithms_names.append('QNWT')

    if 'MQNWT' in args.algs:
        algorithms_names.append('MQNWT')

    if 'LMQNWT' in args.algs:
        algorithms_names.append('LMQNWT')

    if 'ArmijoBasedNSMA' in args.algs:
        algorithms_names.append('ArmijoBasedNSMA')

    if 'WolfeBasedNSMA' in args.algs:
        algorithms_names.append('WolfeBasedNSMA')

    if 'LMQNWTBasedNSMA' in args.algs:
        algorithms_names.append('LMQNWTBasedNSMA')

    if len(algorithms_names) == 0:
        raise Exception('You must insert a set of algorithms')

    front_mode = None
    if ('NWT' in algorithms_names or 'QNWT' in algorithms_names or 'MQNWT' in algorithms_names or 'LMQNWT' in algorithms_names) and ('ArmijoBasedNSMA' in algorithms_names or 'WolfeBasedNSMA' in algorithms_names or 'LMQNWTBasedNSMA' in algorithms_names):
        raise Exception('The employment of a front-based and a local search algorithms in one execution is not allowed')
    elif 'NWT' in algorithms_names or 'QNWT' in algorithms_names or 'MQNWT' in algorithms_names or 'LMQNWT' in algorithms_names:
        front_mode = False
    elif 'ArmijoBasedNSMA' in algorithms_names or 'WolfeBasedNSMA' in algorithms_names or 'LMQNWTBasedNSMA' in algorithms_names:
        front_mode = True

    problems = []
    n_problems = 0

    if 'CEC' in args.probs:
        problems.extend(PROBLEMS['CEC'])
        for problem in PROBLEMS['CEC']:
            n_problems += len(PROBLEM_DIMENSIONS[problem.family_name()])
    
    if 'JOS' in args.probs:
        problems.extend(PROBLEMS['JOS'])
        for problem in PROBLEMS['JOS']:
            n_problems += len(PROBLEM_DIMENSIONS[problem.family_name()])

    if 'MFDS' in args.probs:
        problems.extend(PROBLEMS['MFDS'])
        for problem in PROBLEMS['MFDS']:
            n_problems += len(PROBLEM_DIMENSIONS[problem.family_name()])

    if 'MAN' in args.probs:
        problems.extend(PROBLEMS['MAN'])
        for problem in PROBLEMS['MAN']:
            n_problems += len(PROBLEM_DIMENSIONS[problem.family_name()])

    if 'MMR' in args.probs:
        problems.extend(PROBLEMS['MMR'])
        for problem in PROBLEMS['MMR']:
            n_problems += len(PROBLEM_DIMENSIONS[problem.family_name()])

    if 'MOP' in args.probs:
        problems.extend(PROBLEMS['MOP'])
        for problem in PROBLEMS['MOP']:
            n_problems += len(PROBLEM_DIMENSIONS[problem.family_name()])

    if 'MOP7' in args.probs:
        problems.extend(PROBLEMS['MOP7'])
        for problem in PROBLEMS['MOP7']:
            n_problems += len(PROBLEM_DIMENSIONS[problem.family_name()])

    if 'SLC' in args.probs:
        problems.extend(PROBLEMS['SLC'])
        for problem in PROBLEMS['SLC']:
            n_problems += len(PROBLEM_DIMENSIONS[problem.family_name()])

    if len(problems) == 0:
        raise Exception('You must insert a set of test problems')

    general_settings = {'seeds': args.seeds,
                        'num_trials': args.num_trials,
                        'max_iter': args.max_iter,
                        'max_time': args.max_time,
                        'max_f_evals': args.max_f_evals,
                        'verbose': args.verbose,
                        'verbose_interspace': args.verbose_interspace,
                        'plot_pareto_front': args.plot_pareto_front,
                        'plot_pareto_solutions': args.plot_pareto_solutions,
                        'general_export': args.general_export,
                        'export_pareto_solutions': args.export_pareto_solutions,
                        'plot_dpi': args.plot_dpi}

    NWT_settings = {'theta_tol': args.NWT_theta_for_stationarity}

    QNWT_settings = {'theta_tol': args.QNWT_theta_for_stationarity}

    MQNWT_settings = {'theta_tol': args.MQNWT_theta_for_stationarity}

    LMQNWT_settings = {'max_cor': args.LMQNWT_max_cor,
                       'theta_tol': args.LMQNWT_theta_for_stationarity}

    NSMA_settings = {'pop_size': args.NSMA_pop_size,
                     'crossover_probability': args.NSMA_crossover_probability,
                     'crossover_eta': args.NSMA_crossover_eta,
                     'mutation_eta': args.NSMA_mutation_eta,
                     'shift': args.NSMA_shift,
                     'crowding_quantile': args.NSMA_crowding_quantile,
                     'n_opt': args.NSMA_n_opt,
                     'FMOPG_max_iter': args.NSMA_FMOPG_max_iter,
                     'theta_for_stationarity': args.NSMA_theta_for_stationarity,
                     'theta_tol': args.NSMA_theta_tol,
                     'theta_dec_factor': args.NSMA_theta_dec_factor}

    LMQNWTBasedNSMA_settings = {'pop_size': args.NSMA_pop_size,
                                'crossover_probability': args.NSMA_crossover_probability,
                                'crossover_eta': args.NSMA_crossover_eta,
                                'mutation_eta': args.NSMA_mutation_eta,
                                'shift': args.NSMA_shift,
                                'crowding_quantile': args.NSMA_crowding_quantile,
                                'n_opt': args.NSMA_n_opt,
                                'LMQNWT_max_iter': args.LMQNWTBasedNSMA_LMQNWT_max_iter,
                                'LMQNWT_max_cor': args.LMQNWTBasedNSMA_LMQNWT_max_cor,
                                'theta_for_stationarity': args.NSMA_theta_for_stationarity,
                                'theta_tol': args.NSMA_theta_tol,
                                'theta_dec_factor': args.NSMA_theta_dec_factor}

    algorithms_settings = {'NWT': NWT_settings,
                           'QNWT': QNWT_settings,
                           'MQNWT': MQNWT_settings,
                           'LMQNWT': LMQNWT_settings,
                           'ArmijoBasedNSMA': NSMA_settings,
                           'WolfeBasedNSMA': NSMA_settings,
                           'LMQNWTBasedNSMA': LMQNWTBasedNSMA_settings}

    DDS_settings = {'gurobi_method': args.gurobi_method,
                    'gurobi_verbose': args.gurobi_verbose}

    ALS_settings = {'ALS_alpha_0': args.ALS_alpha_0,
                    'ALS_delta': args.ALS_delta,
                    'ALS_beta': args.ALS_beta,
                    'ALS_min_alpha': args.ALS_min_alpha}

    WLS_settings = {'WLS_alpha_0': args.WLS_alpha_0,
                    'WLS_beta': args.WLS_beta,
                    'WLS_sigma': args.WLS_sigma,
                    'WLS_tau': args.WLS_tau,
                    'WLS_gamma': args.WLS_gamma,
                    'WLS_min_alpha': args.WLS_min_alpha,
                    'WLS_max_alpha': args.WLS_max_alpha}

    return algorithms_names, problems, n_problems, front_mode, general_settings, algorithms_settings, DDS_settings, ALS_settings, WLS_settings


def args_file_creation(seed: int, date: str, args):
    if args.general_export:
        args_file = open(os.path.join('Execution_Outputs', date, str(seed), 'params.csv'), 'w')
        for key in args.__dict__.keys():
            if type(args.__dict__[key]) == float:
                args_file.write('{};{}\n'.format(key, str(round(args.__dict__[key], 10)).replace('.', ',')))
            else:
                args_file.write('{};{}\n'.format(key, args.__dict__[key]))
        args_file.close()


