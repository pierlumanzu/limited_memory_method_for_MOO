import numpy as np
import tensorflow as tf

from problems.jos.jos_class import JOS

'''
For more details about the JOS problems, the user is referred to 

Jin, Y., Olhofer, M., Sendhoff, B.: Dynamic weighted aggregation for evo-
lutionary multi-objective optimization: Why does it work and how? In:
Proceedings of the Genetic and Evolutionary Computation Conference,
pp. 1042–1049 (2001).
'''

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
