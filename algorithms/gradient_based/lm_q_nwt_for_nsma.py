import numpy as np

from nsma.algorithms.gradient_based.local_search_algorithms.fmopg import FMOPG

from algorithms.gradient_based.lm_q_nwt import LMQNWT
from problems.extended_problem import ExtendedProblem


class LMQNWTForNSMA(LMQNWT, FMOPG):

    def __init__(self,
                 theta_tol: float,
                 max_cor: int,
                 gurobi_method: int,
                 gurobi_verbose: bool,
                 args_line_search: dict,
                 max_iter: int = None,
                 max_time: float = None,
                 max_f_evals: int = None):

        LMQNWT.__init__(self,
                        max_iter, max_time, max_f_evals,
                        False, 0,
                        False, False, 0,
                        theta_tol, max_cor,
                        gurobi_method, gurobi_verbose,
                        args_line_search)

        self.__theta_array = np.array([-np.inf], dtype=float)
        LMQNWT.add_stopping_condition(self, 'theta_tolerance', theta_tol, self.__theta_array[0], equal_required=True)

        self.__alpha_array = np.array([1], dtype=float)
        LMQNWT.add_stopping_condition(self, 'min_alpha', 0, self.__alpha_array[0], smaller_value_required=True, equal_required=True)

    def search(self, p_list: np.array, f_list: np.array, problem: ExtendedProblem, index_initial_point: int = None, I: tuple = None):

        n_current_points, n = p_list.shape
        m = f_list.shape[1]

        p_list_tmp = p_list[index_initial_point, :].reshape(1, n)
        f_list_tmp = f_list[index_initial_point, :].reshape(1, m)

        optimization_success = False

        J = problem.evaluate_functions_jacobian(p_list_tmp[0, :])
        self.add_to_stopping_condition_current_value('max_f_evals', n)

        HG_results = self.hg_procedure(J)

        while not self.evaluate_stopping_conditions():

            n_iteration = self.get_stopping_condition_current_value('max_iter')

            v, theta, lam, reset = self._direction_solver.compute_direction(problem, J, None, HG_results)
            self.__theta_array[n_iteration] = theta
            self.update_stopping_condition_current_value('theta_tolerance', self.__theta_array[n_iteration])

            if (np.max(np.dot(J, v), axis=0) >= 0 or theta >= self._theta_tol) and not reset:
                v, theta, lam = self._local_procedure_for_termination.compute_direction(problem, J, None)
                self.__theta_array[n_iteration] = theta
                self.update_stopping_condition_current_value('theta_tolerance', self.__theta_array[n_iteration])

                reset = True
                p_list_tmp, f_list_tmp, n_iteration = self.change_iteration(p_list_tmp, f_list_tmp, n_iteration)
                if self.evaluate_stopping_conditions():
                    break

            if reset:
                self.reset_correction_vectors()

            if theta < self._theta_tol:

                exit_line = False

                while not exit_line:

                    new_p, new_f, alpha, f_eval_ls = self._line_search.search(problem, p_list_tmp[n_iteration, :], f_list_tmp[n_iteration, :], v, np.max(np.dot(J, v), axis=0))
                    self.add_to_stopping_condition_current_value('max_f_evals', f_eval_ls)

                    self.__alpha_array[n_iteration] = alpha
                    self.update_stopping_condition_current_value('min_alpha', self.__alpha_array[n_iteration])

                    if alpha != 0:
                        new_J = problem.evaluate_functions_jacobian(new_p)
                        self.add_to_stopping_condition_current_value('max_f_evals', n)

                        if self.evaluate_stopping_conditions():
                            break

                        self.update_memory_vectors(p_list_tmp[n_iteration, :], new_p, J, new_J, lam)
                        J = new_J
                        HG_results = self.hg_procedure(J)

                        optimization_success = True

                        p_list_tmp = np.concatenate((p_list_tmp, new_p.reshape((1, n))), axis=0)
                        f_list_tmp = np.concatenate((f_list_tmp, new_f.reshape((1, m))), axis=0)

                        self.__theta_array = np.concatenate((self.__theta_array, np.array([-np.inf])), axis=0)
                        self.update_stopping_condition_current_value('theta_tolerance', self.__theta_array[n_iteration + 1])

                        self.__alpha_array = np.concatenate((self.__alpha_array, np.array([1])), axis=0)
                        self.update_stopping_condition_current_value('min_alpha', self.__alpha_array[n_iteration + 1])

                        exit_line = True

                    else:
                        if reset:
                            exit_line = True
                        else:
                            v, theta, lam = self._local_procedure_for_termination.compute_direction(problem, J, None)
                            self.__theta_array[n_iteration] = theta
                            self.update_stopping_condition_current_value('theta_tolerance', self.__theta_array[n_iteration])

                            reset = True
                            p_list_tmp, f_list_tmp, n_iteration = self.change_iteration(p_list_tmp, f_list_tmp, n_iteration)
                            if self.evaluate_stopping_conditions():
                                break

                            self.reset_correction_vectors()

            self.add_to_stopping_condition_current_value('max_iter', 1)

        if optimization_success:
            p_list = np.concatenate((p_list, p_list_tmp[-1, :].reshape(1, n)), axis=0)
            f_list = np.concatenate((f_list, f_list_tmp[-1, :].reshape(1, m)), axis=0)

        return p_list, f_list, self.__theta_array

    def reset_stopping_conditions_current_values(self, theta_tol: float):
        self.update_stopping_condition_current_value('max_iter', 0)

        self._theta_tol = theta_tol
        self.__theta_array = np.array([-np.inf], dtype=float)
        self.update_stopping_condition_current_value('theta_tolerance', self.__theta_array[0])
        self.update_stopping_condition_reference_value('theta_tolerance', theta_tol)

        self.__alpha_array = np.array([1], dtype=float)
        self.update_stopping_condition_current_value('min_alpha', self.__alpha_array[0])

        self.reset_correction_vectors()

    def change_iteration(self, p_list_tmp: np.array, f_list_tmp: np.array, n_iteration: int):
        n = p_list_tmp.shape[1]
        m = f_list_tmp.shape[1]

        p_list_tmp = np.concatenate((p_list_tmp, p_list_tmp[-1, :].reshape((1, n))), axis=0)
        f_list_tmp = np.concatenate((f_list_tmp, f_list_tmp[-1, :].reshape((1, m))), axis=0)

        self.__theta_array = np.concatenate((self.__theta_array, np.array([self.__theta_array[-1]])), axis=0)
        self.update_stopping_condition_current_value('theta_tolerance', self.__theta_array[n_iteration + 1])

        self.__alpha_array = np.concatenate((self.__alpha_array, np.array([self.__alpha_array[-1]])), axis=0)
        self.update_stopping_condition_current_value('min_alpha', self.__alpha_array[n_iteration + 1])

        self.add_to_stopping_condition_current_value('max_iter', 1)

        return p_list_tmp, f_list_tmp, self.get_stopping_condition_current_value('max_iter')
