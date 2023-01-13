import numpy as np
import tensorflow as tf

from problems.mmr.mmr_class import MMR

'''
For more details about the MMR5 problem, the user is referred to 

Miglierina, E., Molho, E., Recchioni, M.C.: Box-constrained multi-
objective optimization: A gradient-like method without “a priori” scalar-
ization. European Journal of Operational Research 188(3), 662–682
(2008). https://doi.org/10.1016/j.ejor.2007.05.015.
'''

class MMR5(MMR):

    def __init__(self, n: int):
        assert n >= 1

        MMR.__init__(self, n)

        self.set_objectives([
            (1 / self.n * tf.reduce_sum([self._z[i] ** 2 - 10 * tf.cos(2 * np.pi * self._z[i]) + 10 for i in range(self.n)])) ** 0.25,
            (1 / self.n * tf.reduce_sum([(self._z[i] - 1.5) ** 2 - 10 * tf.cos(2 * np.pi * (self._z[i] - 1.5)) + 10 for i in range(self.n)])) ** 0.25
        ])

        self.filtered_lb_for_ini = -5 * np.ones(self.n)
        self.filtered_ub_for_ini = 5 * np.ones(self.n)

    @staticmethod
    def name():
        return 'MMR5'
