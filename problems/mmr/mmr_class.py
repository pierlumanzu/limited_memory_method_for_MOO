from abc import ABC

from problems.extended_problem import ExtendedProblem

'''
For more details about the MMR5 problem, the user is referred to 

Miglierina, E., Molho, E., Recchioni, M.C.: Box-constrained multi-
objective optimization: A gradient-like method without “a priori” scalar-
ization. European Journal of Operational Research 188(3), 662–682
(2008). https://doi.org/10.1016/j.ejor.2007.05.015.
'''

class MMR(ExtendedProblem, ABC):

    def __init__(self, n: int):
        ExtendedProblem.__init__(self, n)

    @staticmethod
    def family_name():
        return 'MMR'
