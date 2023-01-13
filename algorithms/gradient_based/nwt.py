import numpy as np
import time

from algorithms.gradient_based.extended_gradient_based_algorithm import ExtendedGradientBasedAlgorithm
from direction_solvers.direction_solver_factory import DirectionSolverFactory
from line_searches.line_search_factory import LineSearchFactory
from problems.extended_problem import ExtendedProblem


class NWT(ExtendedGradientBasedAlgorithm):

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

        self._direction_solver = DirectionSolverFactory.get_direction_calculator('NWTDirection', verbose, gurobi_method, gurobi_verbose)
        self._line_search = LineSearchFactory.get_line_search('MOWLS', args_line_search)

        self.__local_procedure_for_termination = DirectionSolverFactory.get_direction_calculator('SteepestDirection', verbose, gurobi_method, gurobi_verbose)

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

            Hes = problem.evaluate_functions_hessian(p_list[i, :])
            n_h_evals[i] += 1

            if self.evaluate_stopping_conditions():
                enter_while = False

            while not self.evaluate_stopping_conditions() and self._theta < self._theta_tol and self._alpha != 0 and enter_while:

                self.add_to_stopping_condition_current_value('max_iter', 1)

                v, self._theta, reset = self._direction_solver.compute_direction(problem, J, None, Hes)

                if self.evaluate_stopping_conditions():
                    break

                if (np.max(np.dot(J, v), axis=0) >= 0 or self._theta >= self._theta_tol) and not reset:
                    v, self._theta, _ = self.__local_procedure_for_termination.compute_direction(problem, J, None)

                    if self.evaluate_stopping_conditions():
                        break

                    reset = True
                    self.add_to_stopping_condition_current_value('max_iter', 1)

                if self._theta < self._theta_tol:

                    exit_line = False

                    while not exit_line:

                        new_p, new_f, self._alpha, f_eval_ls = self._line_search.search(problem, p_list[i, :], f_list[i, :], v, np.max(np.dot(J, v), axis=0))
                        self.add_to_stopping_condition_current_value('max_f_evals', f_eval_ls)

                        if self.evaluate_stopping_conditions():
                            break

                        if self._alpha != 0:
                            p_list[i, :] = new_p
                            f_list[i, :] = new_f

                            J = problem.evaluate_functions_jacobian(p_list[i, :])
                            n_g_evals[i] += 1

                            if self.evaluate_stopping_conditions():
                                break

                            Hes = problem.evaluate_functions_hessian(p_list[i, :])
                            n_h_evals[i] += 1

                            if self.evaluate_stopping_conditions():
                                break

                            exit_line = True

                        else:
                            if reset:
                                exit_line = True
                            else:
                                v, self._theta, _ = self.__local_procedure_for_termination.compute_direction(problem, J, None)

                                if self.evaluate_stopping_conditions():
                                    break

                                reset = True
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
