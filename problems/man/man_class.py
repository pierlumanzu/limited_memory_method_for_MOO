from abc import ABC

from problems.extended_problem import ExtendedProblem

'''
For more details about the MAN1 problem, the user is referred to 

Lapucci, M., Mansueto, P., Schoen, F.: A memetic procedure for global
multi-objective optimization. Mathematical Programming Computation
(2022). https://doi.org/10.1007/s12532-022-00231-3.
'''

class MAN(ExtendedProblem, ABC):

    def __init__(self, n: int):
        ExtendedProblem.__init__(self, n)

    @staticmethod
    def family_name():
        return 'MAN'
