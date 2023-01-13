import numpy as np
import tensorflow as tf

from problems.cec.cec_class import CEC


class CEC091(CEC):

    def __init__(self, n: int):
        assert n >= 3
        CEC.__init__(self, n)

        J1 = np.arange(2, self.n, 2)
        J2 = np.arange(1, self.n, 2)

        self.set_objectives([
            self._z[0] + (2 / len(J1)) * tf.reduce_sum([(self._z[i] - tf.sin(6 * np.pi * self._z[0] + (i + 1) * np.pi / n)) ** 2 for i in J1]),
            1 - tf.sqrt(self._z[0]) + (2 / len(J2)) * tf.reduce_sum([(self._z[i] - tf.sin(6 * np.pi * self._z[0] + (i + 1) * np.pi / n)) ** 2 for i in J2])
        ])

        filtered_lb_for_ini = -1 * np.ones(n)
        filtered_lb_for_ini[0] = 0.0
        self.filtered_lb_for_ini = filtered_lb_for_ini

        self.filtered_ub_for_ini = np.ones(n)

    @staticmethod
    def name():
        return 'CEC091'


class CEC092(CEC):

    def __init__(self, n: int):
        assert n >= 3
        CEC.__init__(self, n)

        J1 = np.arange(2, self.n, 2)
        J2 = np.arange(1, self.n, 2)

        y_odd = 2 * tf.reduce_sum([(self._z[j] - (0.3 * self._z[0] ** 2 * tf.cos(24 * np.pi * self._z[0] + 4 * (j + 1) * np.pi / n) + 0.6 * self._z[0]) * tf.cos(6 * np.pi * self._z[0] + (j + 1) * np.pi / n)) ** 2 for j in J1]) / len(J1)
        y_even = 2 * tf.reduce_sum([(self._z[j] - (0.3 * self._z[0] ** 2 * tf.cos(24 * np.pi * self._z[0] + 4 * (j + 1) * np.pi / n) + 0.6 * self._z[0]) * tf.sin(6 * np.pi * self._z[0] + (j + 1) * np.pi / n)) ** 2 for j in J2]) / len(J2)

        self.set_objectives([
            self._z[0] + y_odd,
            1 - tf.sqrt(self._z[0]) + y_even
        ])

        filtered_lb_for_ini = -1 * np.ones(n)
        filtered_lb_for_ini[0] = 0.0
        self.filtered_lb_for_ini = filtered_lb_for_ini

        self.filtered_ub_for_ini = np.ones(n)

    @staticmethod
    def name():
        return 'CEC092'


class CEC093(CEC):

    def __init__(self, n: int):
        assert n >= 3
        CEC.__init__(self, n)

        J1 = np.arange(2, self.n, 2)
        J2 = np.arange(1, self.n, 2)

        y = [self._z[j] - self._z[0] ** (0.5 * (1 + 3 * (j - 1) / (n - 2))) for j in np.arange(self.n)]

        y_1 = tf.reduce_sum([y[j] ** 2 for j in J1])
        y_2 = tf.reduce_prod([tf.cos(20 * y[j] * np.pi / np.sqrt(j + 1.0)) for j in J1])

        y_odd = 2 * (4 * y_1 - 2 * y_2 + 2) / len(J1)
        y_even = 2 * (4 * tf.reduce_sum([y[j] ** 2 for j in J2]) - 2 * tf.reduce_prod([tf.cos(20 * y[j] * np.pi / np.sqrt(j + 1.0)) for j in J2]) + 2) / len(J2)

        self.set_objectives([
            self._z[0] + y_odd,
            1 - tf.sqrt(self._z[0]) + y_even
        ])

        self.filtered_lb_for_ini = np.zeros(n)
        self.filtered_ub_for_ini = np.ones(n)

    @staticmethod
    def name():
        return 'CEC093'


class CEC097(CEC):

    def __init__(self, n: int):
        assert n >= 3
        CEC.__init__(self, n)

        J1 = np.arange(2, self.n, 2)
        J2 = np.arange(1, self.n, 2)

        y = [self._z[j] - tf.sin(6 * np.pi * self._z[0] + (j + 1) * np.pi / self.n) for j in np.arange(self.n)]

        self.set_objectives([
            self._z[0] ** (1 / 5) + (2 / len(J1)) * tf.reduce_sum([y[j] ** 2 for j in J1]),
            1 - self._z[0] ** (1 / 5) + (2 / len(J2)) * tf.reduce_sum([y[j] ** 2 for j in J2])
        ])

        filtered_lb_for_ini = -1 * np.ones(n)
        filtered_lb_for_ini[0] = 1.0e-6
        self.filtered_lb_for_ini = filtered_lb_for_ini

        self.filtered_ub_for_ini = np.ones(n)

    @staticmethod
    def name():
        return 'CEC097'


class CEC098(CEC):

    def __init__(self, n: int):
        assert n >= 5
        CEC.__init__(self, n)

        J1 = np.arange(3, self.n, 3)
        J2 = np.arange(4, self.n, 3)
        J3 = np.arange(2, self.n, 3)

        self.set_objectives([
            tf.cos(0.5 * np.pi * self._z[0]) * tf.cos(0.5 * np.pi * self._z[1]) + (2 / len(J1)) * tf.reduce_sum([(self._z[j] - 2 * self._z[1] * tf.sin(2 * np.pi * self._z[0] + (j + 1) * np.pi / self.n)) ** 2 for j in J1]),
            tf.cos(0.5 * np.pi * self._z[0]) * tf.sin(0.5 * np.pi * self._z[1]) + (2 / len(J2)) * tf.reduce_sum([(self._z[j] - 2 * self._z[1] * tf.sin(2 * np.pi * self._z[0] + (j + 1) * np.pi / self.n)) ** 2 for j in J2]),
            tf.sin(0.5 * np.pi * self._z[0]) + (2 / len(J3)) * tf.reduce_sum([(self._z[j] - 2 * self._z[1] * tf.sin(2 * np.pi * self._z[0] + (j + 1) * np.pi / self.n)) ** 2 for j in J3])
        ])

        filtered_lb_for_ini = -2 * np.ones(n)
        filtered_lb_for_ini[0] = 0.0
        filtered_lb_for_ini[1] = 0.0
        self.filtered_lb_for_ini = filtered_lb_for_ini

        filtered_ub_for_ini = 2 * np.ones(n)
        filtered_ub_for_ini[0] = 1.0
        filtered_ub_for_ini[1] = 1.0
        self.filtered_ub_for_ini = filtered_ub_for_ini

    @staticmethod
    def name():
        return 'CEC098'


class CEC0910(CEC):

    def __init__(self, n: int):
        assert n >= 5
        CEC.__init__(self, n)

        J1 = np.arange(3, self.n, 3)
        J2 = np.arange(4, self.n, 3)
        J3 = np.arange(2, self.n, 3)

        y = [self._z[j] - 2 * self._z[1] * tf.sin(2 * np.pi * self._z[0] + (j + 1) * np.pi / self.n) for j in np.arange(self.n)]

        self.set_objectives([
            tf.cos(0.5 * self._z[0] * np.pi) * tf.cos(0.5 * self._z[1] * np.pi) + (2 / len(J1)) * tf.reduce_sum([4 * y[j] ** 2 - tf.cos(8 * np.pi * y[j]) + 1 for j in J1]),
            tf.cos(0.5 * self._z[0] * np.pi) * tf.sin(0.5 * self._z[1] * np.pi) + (2 / len(J2)) * tf.reduce_sum([4 * y[j] ** 2 - tf.cos(8 * np.pi * y[j]) + 1 for j in J2]),
            tf.sin(0.5 * self._z[0] * np.pi) + (2 / len(J3)) * tf.reduce_sum([4 * y[j] ** 2 - tf.cos(8 * np.pi * y[j]) + 1 for j in J3])
        ])

        filtered_lb_for_ini = -2 * np.ones(n)
        filtered_lb_for_ini[0] = 0.0
        filtered_lb_for_ini[1] = 0.0
        self.filtered_lb_for_ini = filtered_lb_for_ini

        filtered_ub_for_ini = 2 * np.ones(n)
        filtered_ub_for_ini[0] = 1.0
        filtered_ub_for_ini[1] = 1.0
        self.filtered_ub_for_ini = filtered_ub_for_ini

    @staticmethod
    def name():
        return 'CEC0910'
