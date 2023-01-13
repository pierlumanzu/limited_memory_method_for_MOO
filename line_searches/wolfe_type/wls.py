from abc import ABC

from nsma.line_searches.armijo_type.als import ALS


class WLS(ALS, ABC):

    def __init__(self, alpha_0: float, beta: float, sigma: float, tau: float, gamma: float, min_alpha: float, max_alpha: float):

        ALS.__init__(self, alpha_0, gamma, beta, min_alpha)

        self._sigma = sigma
        self._tau = tau
        self._max_alpha = max_alpha
