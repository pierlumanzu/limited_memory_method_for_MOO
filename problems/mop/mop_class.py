from abc import ABC

from problems.extended_problem import ExtendedProblem

'''
For more details about the MOP problems, the user is referred to 

Huband, S., Hingston, P., Barone, L., While, L.: A review of multiobjec-
tive test problems and a scalable test problem toolkit. IEEE Transactions
on Evolutionary Computation 10(5), 477–506 (2006). https://doi.org/10.1109/TEVC.2005.861417.
'''

class MOP(ExtendedProblem, ABC):

    def __init__(self, n: int):
        ExtendedProblem.__init__(self, n)

    @staticmethod
    def family_name():
        return 'MOP'
