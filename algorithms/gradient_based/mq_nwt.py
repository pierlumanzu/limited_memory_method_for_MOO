import numpy as np
import time

from algorithms.gradient_based.extended_gradient_based_algorithm import ExtendedGradientBasedAlgorithm
from direction_solvers.direction_solver_factory import DirectionSolverFactory
from line_searches.line_search_factory import LineSearchFactory
from problems.extended_problem import ExtendedProblem


class MQNWT(ExtendedGradientBasedAlgorithm):

    def __init__(self,
                 max_iter: int, max_time: float, max_f_evals: int,
                 verbose: bool, verbose_interspace: int,
                 plot_pareto_front: bool, plot_pareto_solutions: bool, plot_dpi: int,
                 theta_tol: float,
                 gurobi_method: int, gurobi_verbose: bool,
                 args_line_search: dict):

        ExtendedGradientBasedAlgorithm.__init__(self,
                                                max_iter, max_time, max_f_evals,
                                                verbose, verbose_interspace,
                                                plot_pareto_front, plot_pareto_solutions, plot_dpi,
                                                theta_tol,
                                                gurobi_method, gurobi_verbose)

        self._direction_solver = DirectionSolverFactory.get_direction_calculator('MQNWTDirection', verbose, gurobi_method, gurobi_verbose)
        self._line_search = LineSearchFactory.get_line_search('MOWLS', args_line_search)

        self.__local_procedure_for_termination = DirectionSolverFactory.get_direction_calculator('SteepestDirection', verbose, gurobi_method, gurobi_verbose)

        self.__H = None

    def search(self, p_list: np.array, f_list: np.array, problem: ExtendedProblem):

        n_points, n = p_list.shape

        elapsed_time = np.zeros(n_points, dtype=np.float)
        n_iterations = np.zeros(n_points, dtype=np.int)
        n_f_evals = np.zeros(n_points, dtype=np.int)
        n_g_evals = np.zeros(n_points, dtype=np.int)
        n_h_evals = np.zeros(n_points, dtype=np.int)
        thetas = np.zeros(n_points, dtype=np.float)

        self.show_figure(p_list, f_list)

        for i in range(n_points):
            self.reset_all()
            enter_while = True

            J = problem.evaluate_functions_jacobian(p_list[i, :])
            n_g_evals[i] += 1

            if self.evaluate_stopping_conditions():
                enter_while = False

            self.__H = np.eye(n, dtype=np.float)

            while not self.evaluate_stopping_conditions() and self._theta < self._theta_tol and self._alpha != 0 and enter_while:

                self.add_to_stopping_condition_current_value('max_iter', 1)

                v, self._theta, lam, reset = self._direction_solver.compute_direction(problem, J, None, self.__H)

                if self.evaluate_stopping_conditions():
                    continue

                if (np.max(np.dot(J, v), axis=0) >= 0 or self._theta >= self._theta_tol) and not reset:
                    v, self._theta, lam = self.__local_procedure_for_termination.compute_direction(problem, J, None)

                    if self.evaluate_stopping_conditions():
                        break

                    reset = True
                    self.add_to_stopping_condition_current_value('max_iter', 1)

                if reset:
                    self.reset_approx_inv_hes()

                if self._theta < self._theta_tol:

                    exit_line = False

                    while not exit_line:

                        new_p, new_f, self._alpha, f_eval_ls = self._line_search.search(problem, p_list[i, :], f_list[i, :], v, np.max(np.dot(J, v), axis=0))
                        self.add_to_stopping_condition_current_value('max_f_evals', f_eval_ls)

                        if self.evaluate_stopping_conditions():
                            break

                        if self._alpha != 0:
                            new_J = problem.evaluate_functions_jacobian(new_p)
                            n_g_evals[i] += 1

                            if self.evaluate_stopping_conditions():
                                break

                            self.approx_inv_hes(J, p_list[i, :], new_J, new_p, lam)
                            J = new_J

                            p_list[i, :] = new_p
                            f_list[i, :] = new_f

                            exit_line = True

                        else:
                            if reset:
                                exit_line = True
                            else:
                                v, self._theta, lam = self.__local_procedure_for_termination.compute_direction(problem, J, None)

                                if self.evaluate_stopping_conditions():
                                    break

                                reset = True
                                self.reset_approx_inv_hes()
                                self.add_to_stopping_condition_current_value('max_iter', 1)

                    if reset and self._alpha == 0:
                        break

                else:
                    break

                self.show_figure(p_list, f_list)

            elapsed_time[i] = time.time() - self.get_stopping_condition_current_value('max_time')
            n_iterations[i] = self.get_stopping_condition_current_value('max_iter')
            n_f_evals[i] = self.get_stopping_condition_current_value('max_f_evals')

            J = problem.evaluate_functions_jacobian(p_list[i, :])
            _, self._theta, _ = self.__local_procedure_for_termination.compute_direction(problem, J, None)
            thetas[i] = self._theta

        self.close_figure()

        return p_list, f_list, elapsed_time, n_iterations, n_f_evals, n_g_evals, n_h_evals, thetas

    def approx_inv_hes(self, J_ini: np.array, p_ini: np.array, J_fin: np.array, p_fin: np.array, lam: np.array):
        m, n = J_ini.shape

        if self.__H is None:
            self.__H = np.eye(n, dtype=np.float)

        s = p_fin - p_ini
        u = np.dot(lam, J_fin - J_ini)

        sy = np.dot(s, u)

        if sy > 0:
            ro = 1 / sy
        else:
            if self._verbose:
                print('Warning sy <= 0')
            ro = 1 / np.dot((np.max(np.dot(J_fin, s)) * np.ones(m, dtype=np.float) - np.dot(J_ini, s)), lam)

        V = np.eye(n, dtype=np.float) - ro * np.dot(u[:, None], s[None, :])

        self.__H = np.dot(V.T, np.dot(self.__H, V)) + ro * np.dot(s[:, None], s[None, :])

    def reset_all(self):
        ExtendedGradientBasedAlgorithm.reset_all(self)
        self.reset_approx_inv_hes()

    def reset_approx_inv_hes(self):
        self.__H = None
