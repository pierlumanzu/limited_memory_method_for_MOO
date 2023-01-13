from algorithms.gradient_based.wolfe_based_fmopg import WolfeBasedFMOPG
from algorithms.memetic.armijo_based_nsma import ArmijoBasedNSMA


class WolfeBasedNSMA(ArmijoBasedNSMA):

    def __init__(self,
                 max_iter: int,
                 max_time: float,
                 max_f_evals: int,
                 verbose: bool,
                 verbose_interspace: int,
                 plot_pareto_front: bool,
                 plot_pareto_solutions: bool,
                 plot_dpi: int,
                 pop_size: int,
                 crossover_probability: float,
                 crossover_eta: float,
                 mutation_eta: float,
                 shift: float,
                 crowding_quantile: float,
                 n_opt: int,
                 FMOPG_max_iter: int,
                 theta_for_stationarity: float,
                 theta_tol: float,
                 theta_dec_factor: float,
                 gurobi_method: int,
                 gurobi_verbose: bool,
                 args_line_search: dict):

        ArmijoBasedNSMA.__init__(self,
                                 max_iter, max_time, max_f_evals,
                                 verbose, verbose_interspace,
                                 plot_pareto_front, plot_pareto_solutions, plot_dpi,
                                 pop_size,
                                 crossover_probability, crossover_eta, mutation_eta,
                                 shift, crowding_quantile,
                                 n_opt, 0,
                                 theta_for_stationarity, theta_tol, theta_dec_factor,
                                 gurobi_method, gurobi_verbose,
                                 {'ALS_alpha_0': 0, 'ALS_delta': 0, 'ALS_beta': 0, 'ALS_min_alpha': 0})

        self._local_search_optimizer = WolfeBasedFMOPG(theta_tol, gurobi_method, gurobi_verbose, args_line_search, FMOPG_max_iter, max_time, max_f_evals)
