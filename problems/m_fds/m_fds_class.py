from abc import ABC

from problems.extended_problem import ExtendedProblem

'''
For more details about the FDS problem, the user is referred to 

Fliege, J., Drummond, L.G., Svaiter, B.F.: Newton’s method for multi-
objective optimization. SIAM Journal on Optimization 20(2), 602–626
(2009). https://doi.org/10.1137/08071692X.
'''

class MFDS(ExtendedProblem, ABC):

    def __init__(self, n: int):
        ExtendedProblem.__init__(self, n)

    @staticmethod
    def family_name():
        return 'MFDS'
