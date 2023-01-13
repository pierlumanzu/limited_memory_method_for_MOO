import numpy as np
import tensorflow as tf

from problems.jos.jos_class import JOS


class JOS1a(JOS):

    def __init__(self, n: int):
        assert n >= 1

        JOS.__init__(self, n)

        self.set_objectives([
            tf.reduce_sum([self._z[i] ** 2 for i in range(self.n)]) / self.n,
            tf.reduce_sum([(self._z[i] - 2) ** 2 for i in range(self.n)]) / self.n
        ])

        self.filtered_lb_for_ini = -10 * np.ones(self.n)
        self.filtered_ub_for_ini = 10 * np.ones(self.n)

    @staticmethod
    def name():
        return 'JOS1a'


class JOS1b(JOS):

    def __init__(self, n: int):
        assert n >= 1

        JOS.__init__(self, n)

        self.set_objectives([
            tf.reduce_sum([self._z[i] ** 2 for i in range(self.n)]) / self.n,
            tf.reduce_sum([(self._z[i] - 2) ** 2 for i in range(self.n)]) / self.n
        ])

        self.filtered_lb_for_ini = -100 * np.ones(self.n)
        self.filtered_ub_for_ini = 100 * np.ones(self.n)

    @staticmethod
    def name():
        return 'JOS1b'


class JOS1c(JOS):

    def __init__(self, n: int):
        assert n >= 1

        JOS.__init__(self, n)

        self.set_objectives([
            tf.reduce_sum([self._z[i] ** 2 for i in range(self.n)]) / self.n,
            tf.reduce_sum([(self._z[i] - 2) ** 2 for i in range(self.n)]) / self.n
        ])

        self.filtered_lb_for_ini = 1e-2 * np.ones(self.n)
        self.filtered_ub_for_ini = 1 * np.ones(self.n)

    @staticmethod
    def name():
        return 'JOS1c'
