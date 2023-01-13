from abc import ABC

from problems.extended_problem import ExtendedProblem


class CEC(ExtendedProblem, ABC):

    def __init__(self, n: int):
        ExtendedProblem.__init__(self, n)

    @staticmethod
    def family_name():
        return 'CEC'
