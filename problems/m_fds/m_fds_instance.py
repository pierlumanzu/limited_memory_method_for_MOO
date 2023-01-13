import numpy as np
import tensorflow as tf

from problems.m_fds.m_fds_class import MFDS

'''
For more details about the FDS problem, the user is referred to 

Fliege, J., Drummond, L.G., Svaiter, B.F.: Newton’s method for multi-
objective optimization. SIAM Journal on Optimization 20(2), 602–626
(2009). https://doi.org/10.1137/08071692X.
'''

class MFDS1(MFDS):

    def __init__(self, n: int):
        assert n >= 1

        MFDS.__init__(self, n)

        self.set_objectives([
            tf.reduce_sum([(i + 1) * (self._z[i] - (i + 1)) ** 4 for i in range(self.n)]) / self.n ** 4,
            tf.exp(tf.reduce_sum([self._z[i] for i in range(self.n)]) / self.n) + tf.norm(self._z) ** 2,
            tf.reduce_sum([(i + 1) * (self.n - i) * tf.exp(-self._z[i]) for i in range(self.n)]) / (self.n * (self.n + 1))
        ])

        self.filtered_lb_for_ini = -2 * np.ones(self.n)
        self.filtered_ub_for_ini = 2 * np.ones(self.n)

    @staticmethod
    def name():
        return 'MFDS1'
