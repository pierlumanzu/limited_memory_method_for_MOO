import numpy as np
import time

from algorithms.gradient_based.extended_gradient_based_algorithm import ExtendedGradientBasedAlgorithm
from direction_solvers.direction_solver_factory import DirectionSolverFactory
from line_searches.line_search_factory import LineSearchFactory
from problems.extended_problem import ExtendedProblem


class LMQNWT(ExtendedGradientBasedAlgorithm):

    def __init__(self,
                 max_iter: int, max_time: float, max_f_evals: int,
                 verbose: bool, verbose_interspace: int,
                 plot_pareto_front: bool, plot_pareto_solutions: bool, plot_dpi: int,
                 theta_tol: float,
                 max_cor: int,
                 gurobi_method: int, gurobi_verbose: bool,
                 args_line_search: dict):

        ExtendedGradientBasedAlgorithm.__init__(self,
                                                max_iter, max_time, max_f_evals,
                                                verbose, verbose_interspace,
                                                plot_pareto_front, plot_pareto_solutions, plot_dpi,
                                                theta_tol,
                                                gurobi_method, gurobi_verbose)

        self._direction_solver = DirectionSolverFactory.get_direction_calculator('LMQNWTDirection', verbose, gurobi_method, gurobi_verbose)
        self._line_search = LineSearchFactory.get_line_search('MOWLS', args_line_search)

        self._local_procedure_for_termination = DirectionSolverFactory.get_direction_calculator('SteepestDirection', verbose, gurobi_method, gurobi_verbose)

        self._max_cor = max_cor

        self._s = None
        self._u = None
        self._ro = None
        self._correction_vector_index = 0

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

            HG_results = self.hg_procedure(J)

            while not self.evaluate_stopping_conditions() and self._theta < self._theta_tol and self._alpha != 0 and enter_while:

                self.add_to_stopping_condition_current_value('max_iter', 1)

                v, self._theta, lam, reset = self._direction_solver.compute_direction(problem, J, None, HG_results)

                if self.evaluate_stopping_conditions():
                    break

                if (np.max(np.dot(J, v), axis=0) >= 0 or self._theta >= self._theta_tol) and not reset:
                    v, self._theta, lam = self._local_procedure_for_termination.compute_direction(problem, J, None)

                    if self.evaluate_stopping_conditions():
                        break

                    reset = True
                    self.add_to_stopping_condition_current_value('max_iter', 1)

                if reset:
                    self.reset_correction_vectors()

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

                            self.update_memory_vectors(p_list[i, :], new_p, J, new_J, lam)
                            J = new_J
                            HG_results = self.hg_procedure(J)

                            p_list[i, :] = new_p
                            f_list[i, :] = new_f

                            exit_line = True

                        else:
                            if reset:
                                exit_line = True
                            else:
                                v, self._theta, lam = self._local_procedure_for_termination.compute_direction(problem, J, None)

                                if self.evaluate_stopping_conditions():
                                    break

                                reset = True
                                self.reset_correction_vectors()
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
            _, self._theta, _ = self._local_procedure_for_termination.compute_direction(problem, J, None)
            thetas[i] = self._theta

        self.close_figure()

        return p_list, f_list, elapsed_time, n_iterations, n_f_evals, n_g_evals, n_h_evals, thetas

    def hg_procedure(self, Jac: np.array):
        m, n = Jac.shape
        n_current_memory_vectors = min(self._correction_vector_index, self._max_cor)

        q = Jac

        alpha = np.empty((m, n_current_memory_vectors), dtype=np.float)

        for i in range(n_current_memory_vectors - 1, -1, -1):
            alpha[:, i] = self._ro[i] * np.dot(q, self._s[:, i])
            q = q - np.dot(alpha[:, i][:, None], self._u[:, i][None, :])

        r = np.dot(np.array([1.]) * np.eye(n), q.T)

        for i in range(n_current_memory_vectors):
            beta = self._ro[i] * np.dot(self._u[:, i][None, :], r)
            r = r + (np.dot(self._s[:, i][:, None], alpha[:, i][None, :] - beta))

        return r

    def update_memory_vectors(self, p_ini: np.array, p_fin: np.array, J_ini: np.array, J_fin: np.array, lam: np.array):
        n = len(p_ini)

        if self._s is None:
            assert self._u is None and self._ro is None

            self._s = np.empty((n, self._max_cor), dtype=np.float)
            self._u = np.empty((n, self._max_cor), dtype=np.float)
            self._ro = np.empty(self._max_cor, dtype=np.float)

        elif self._correction_vector_index >= self._max_cor:
            self._s = np.roll(self._s, -1, axis=1)
            self._u = np.roll(self._u, -1, axis=1)
            self._ro = np.roll(self._ro, -1)

        actual_correction_vector_index = min(self._correction_vector_index, self._max_cor - 1)

        self._s[:, actual_correction_vector_index] = p_fin - p_ini
        self._u[:, actual_correction_vector_index] = np.dot(lam, J_fin - J_ini)

        ro_tmp_den = np.dot(self._u[:, actual_correction_vector_index], self._s[:, actual_correction_vector_index])
        if ro_tmp_den <= 0:
            if self._verbose:
                print('Warning sy <= 0')
            self._u[:, actual_correction_vector_index] = np.dot(lam, J_fin[np.argmax(np.dot(J_fin, self._s[:, actual_correction_vector_index])), :] - J_ini)
            ro_tmp_den = np.dot(self._u[:, actual_correction_vector_index], self._s[:, actual_correction_vector_index])
        self._ro[actual_correction_vector_index] = 1 / ro_tmp_den

        self._correction_vector_index += 1

    def reset_all(self):
        ExtendedGradientBasedAlgorithm.reset_all(self)
        self.reset_correction_vectors()

    def reset_correction_vectors(self):
        self._s = None
        self._u = None
        self._ro = None
        self._correction_vector_index = 0
