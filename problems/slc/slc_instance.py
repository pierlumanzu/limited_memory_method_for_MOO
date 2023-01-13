import numpy as np
import tensorflow as tf

from problems.slc.slc_class import SLC


class SLC2(SLC):

    def __init__(self, n: int):
        assert n >= 2

        SLC.__init__(self, n)

        self.__a1 = np.array([1. for i in range(self.n)])
        self.__a2 = np.array([-1. for i in range(self.n)])

        self.set_objectives([
            tf.reduce_sum([(self._z[i] - self.__a1[i]) ** 2 for i in range(self.n) if i != 0]) + (self._z[0] - self.__a1[0]) ** 4,
            tf.reduce_sum([(self._z[i] - self.__a2[i]) ** 2 for i in range(self.n) if i != 1]) + (self._z[1] - self.__a2[1]) ** 4
        ])

        self.filtered_lb_for_ini = -100 * np.ones(self.n)
        self.filtered_ub_for_ini = 100 * np.ones(self.n)

    @staticmethod
    def name():
        return 'SLC2'

    @staticmethod
    def family_name():
        return 'SLC2'