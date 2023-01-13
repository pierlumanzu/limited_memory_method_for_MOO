from abc import ABC
import numpy as np
import time

from nsma.algorithms.gradient_based.gradient_based_algorithm import GradientBasedAlgorithm


class ExtendedGradientBasedAlgorithm(GradientBasedAlgorithm, ABC):

    def __init__(self,
                 max_iter: int, max_time: float, max_f_evals: int,
                 verbose: bool, verbose_interspace: int,
                 plot_pareto_front: bool, plot_pareto_solutions: bool, plot_dpi: int,
                 theta_tol: float,
                 gurobi_method: int, gurobi_verbose: bool):

        GradientBasedAlgorithm.__init__(self,
                                        max_iter, max_time, max_f_evals,
                                        verbose, verbose_interspace,
                                        plot_pareto_front, plot_pareto_solutions, plot_dpi,
                                        theta_tol,
                                        True, gurobi_method, gurobi_verbose,
                                        0., 0., 0., 0.)

        self._theta = -np.inf
        self._alpha = 1

    def reset_time(self):
        self.update_stopping_condition_current_value('max_time', time.time())

    def reset_all(self):
        self.reset_time()

        self.update_stopping_condition_current_value('max_iter', 0)
        self.update_stopping_condition_current_value('max_f_evals', 0)

        self._theta = -np.inf
        self._alpha = 1