import numpy as np
import tensorflow as tf

from problems.mop.mop_class import MOP


class MMOP2(MOP):

    def __init__(self, n: int):
        assert n >= 1
        MOP.__init__(self, n)

        self.set_objectives([
            1 - tf.exp(-tf.reduce_sum([(self._z[i] - 1 / (self.n ** (1 / 2))) ** 2 for i in range(self.n)]) / self.n),
            1 - tf.exp(-tf.reduce_sum([(self._z[i] + 1 / (self.n ** (1 / 2))) ** 2 for i in range(self.n)]) / self.n)
        ])

        self.filtered_lb_for_ini = -4 * np.ones(n)
        self.filtered_ub_for_ini = 4 * np.ones(n)

    @staticmethod
    def name():
        return 'MMOP2'

    @staticmethod
    def family_name():
        return 'MMOP2'


class MOP3(MOP):

    def __init__(self, n: int):
        assert n == 2
        MOP.__init__(self, n)

        A1 = tf.cast(0.5 * tf.sin(1.0) - 2 * tf.cos(1.0) + tf.sin(2.0) - 1.5 * tf.cos(2.0), dtype=tf.double)
        A2 = tf.cast(1.5 * tf.sin(1.0) - tf.cos(1.0) + 2 * tf.sin(2.0) - 0.5 * tf.cos(2.0), dtype=tf.double)

        B1 = 0.5 * tf.sin(self._z[0]) - 2 * tf.cos(self._z[0]) + tf.sin(self._z[1]) - 1.5 * tf.cos(self._z[1])
        B2 = 1.5 * tf.sin(self._z[0]) - 1 * tf.cos(self._z[0]) + 2 * tf.sin(self._z[1]) - 0.5 * tf.cos(self._z[1])

        self.set_objectives([
            1.0 + (A1 - B1) ** 2 + (A2 - B2) ** 2,
            (self._z[0] + 3) ** 2 + (self._z[1] + 1) ** 2
        ])

        self.filtered_lb_for_ini = -np.pi * np.ones(n)
        self.filtered_ub_for_ini = np.pi * np.ones(n)

    @staticmethod
    def name():
        return 'MOP3'

    @staticmethod
    def family_name():
        return 'MOP_3_7'


class MOP7(MOP):

    def __init__(self, n: int):
        assert n == 2
        MOP.__init__(self, n)

        self.set_objectives([
            ((self._z[0] - 2) ** 2) / 2 + ((self._z[1] + 1) ** 2) / 13 + 3,
            ((self._z[0] + self._z[1] - 3) ** 2) / 36 + ((-self._z[0] + self._z[1] + 2) ** 2) / 8 - 17,
            ((self._z[0] + 2 * self._z[1] - 1) ** 2) / 175 + ((-self._z[0] + 2 * self._z[1]) ** 2) / 17 - 13
        ])

        self.filtered_lb_for_ini = -400 * np.ones(n)
        self.filtered_ub_for_ini = 400 * np.ones(n)

    @staticmethod
    def name():
        return 'MOP7'

    @staticmethod
    def family_name():
        return 'MOP_3_7'
