from abc import ABC

from problems.extended_problem import ExtendedProblem

'''
For more details about the SLC2 problem, the user is referred to 

Schütze, O., Lara, A., Coello, C.C.: The directed search method for
unconstrained multi-objective optimization problems. Proceedings of the
EVOLVE–A Bridge Between Probability, Set Oriented Numerics, and
Evolutionary Computation, 1–4 (2011).
'''

class SLC(ExtendedProblem, ABC):

    def __init__(self, n: int):
        ExtendedProblem.__init__(self, n)

    @staticmethod
    def family_name():
        return 'SLC'
