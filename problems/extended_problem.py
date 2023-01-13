from abc import ABC
import numpy as np
import tensorflow as tf

from nsma.problems.problem import Problem


class ExtendedProblem(Problem, ABC):

    def __init__(self, n: int):
        Problem.__init__(self, n)

        self.__objectives_hessians = np.empty(0)

        self.__filtered_lb_for_ini = np.array([-2.0e19] * self.n, dtype=float)
        self.__filtered_ub_for_ini = np.array([2.0e19] * self.n, dtype=float)

    def evaluate_functions_hessian(self, x: np.array):

        hessian = np.zeros((self.m, self.n, self.n))
        for i in range(self.m):
            if self.__objectives_hessians[i] is not None:
                hessian[i, :, :] = self.__objectives_hessians[i].eval({self._z: x})

        return hessian

    def generate_feasible_random_point(self):
        return np.random.uniform(self.__filtered_lb_for_ini, self.__filtered_ub_for_ini)

    def generate_feasible_points_array(self, mod: str, size: int):
        assert size > 0
        assert mod in ['rand', 'hyper']

        if mod.lower() == 'rand':
            p_list = np.zeros((size, self.n), dtype=float)
            for i in range(size):
                p_list[i, :] = self.generate_feasible_random_point()
        else:
            scale = self.__filtered_ub_for_ini - self.__filtered_lb_for_ini - 2e-3
            shift = self.__filtered_lb_for_ini + 1e-3

            p_list = np.zeros((size, self.n), dtype=float)
            for i in range(size):
                p_list[i, :] = shift + ((i / (size - 1)) if size > 1 else 0.5) * scale

        return p_list

    def set_objectives(self, objectives: list):
        self.objectives = objectives
        self.__objectives_hessians = [tf.hessians(obj, self._z)[0] for obj in objectives]

    @property
    def filtered_lb_for_ini(self):
        return self.__filtered_lb_for_ini

    @filtered_lb_for_ini.setter
    def filtered_lb_for_ini(self, filtered_lb_for_ini: np.array):
        assert len(filtered_lb_for_ini) == self.n
        assert not np.isnan(np.sum(filtered_lb_for_ini))
        assert (filtered_lb_for_ini != np.inf).all()

        self.__filtered_lb_for_ini = filtered_lb_for_ini

    @property
    def filtered_ub_for_ini(self):
        return self.__filtered_ub_for_ini

    @filtered_ub_for_ini.setter
    def filtered_ub_for_ini(self, filtered_ub_for_ini: np.array):
        assert len(filtered_ub_for_ini) == self.n
        assert not np.isnan(np.sum(filtered_ub_for_ini))
        assert (filtered_ub_for_ini != np.inf).all()

        self.__filtered_ub_for_ini = filtered_ub_for_ini
