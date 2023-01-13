from abc import ABC

from problems.extended_problem import ExtendedProblem

'''
For more details about the CEC problems, the user is referred to 

Zhang, Q., Zhou, A., Zhao, S., Suganthan, P.N., Liu, W., Tiwari, S., et
al.: Multiobjective optimization test instances for the cec 2009 special ses-
sion and competition. University of Essex, Colchester, UK and Nanyang
technological University, Singapore, special session on performance assess-
ment of multi-objective optimization algorithms, technical report 264,
1–30 (2008).
'''

class CEC(ExtendedProblem, ABC):

    def __init__(self, n: int):
        ExtendedProblem.__init__(self, n)

    @staticmethod
    def family_name():
        return 'CEC'
