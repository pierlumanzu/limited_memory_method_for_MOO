from abc import ABC

from problems.extended_problem import ExtendedProblem

'''
For more details about the JOS problems, the user is referred to 

Jin, Y., Olhofer, M., Sendhoff, B.: Dynamic weighted aggregation for evo-
lutionary multi-objective optimization: Why does it work and how? In:
Proceedings of the Genetic and Evolutionary Computation Conference,
pp. 1042–1049 (2001).
'''

class JOS(ExtendedProblem, ABC):

    def __init__(self, n: int):
        ExtendedProblem.__init__(self, n)

    @staticmethod
    def family_name():
        return 'JOS'
