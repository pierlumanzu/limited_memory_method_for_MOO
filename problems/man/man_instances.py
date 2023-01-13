import numpy as np
import tensorflow as tf

from problems.man.man_class import MAN

'''
For more details about the MAN1 problem, the user is referred to 

Lapucci, M., Mansueto, P., Schoen, F.: A memetic procedure for global
multi-objective optimization. Mathematical Programming Computation
(2022). https://doi.org/10.1007/s12532-022-00231-3.
'''

class MMAN1(MAN):

    def __init__(self, n: int):
        assert n >= 2

        MAN.__init__(self, n)

        self.set_objectives([
            tf.reduce_sum([(self._z[i] - (i + 1)) ** 2 for i in range(self.n)]) / self.n,
            tf.reduce_sum([tf.exp(-self._z[i]) + self._z[i] for i in range(self.n)])
        ])

        self.filtered_lb_for_ini = -10 * np.ones(self.n)
        self.filtered_ub_for_ini = 10 * np.ones(self.n)

    @staticmethod
    def name():
        return 'MMAN1'


class MAN2(MAN):

    def __init__(self, n: int):
        assert n >= 1

        MAN.__init__(self, n)

        self.set_objectives([
            tf.reduce_sum([(i + 1) * (self._z[i] - (i + 1)) ** 2 for i in range(self.n)]) / self.n ** 2,
            tf.reduce_sum([tf.exp(-self._z[i]) + self._z[i] for i in range(self.n)]),
            tf.reduce_sum([tf.exp(self._z[i] ** 2) for i in range(self.n)])
        ])

        self.filtered_lb_for_ini = -1 * np.ones(self.n)
        self.filtered_ub_for_ini = 1 * np.ones(self.n)

    @staticmethod
    def name():
        return 'MAN2'
