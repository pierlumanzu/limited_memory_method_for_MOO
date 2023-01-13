import numpy as np

from line_searches.wolfe_type.wls import WLS
from line_searches.line_search_utils.pareto_utils import is_dominating_point
from problems.extended_problem import ExtendedProblem


class MOWLS(WLS):

    def __init__(self, alpha_0: float, beta: float, sigma: float, tau: float, gamma: float, min_alpha: float, max_alpha: float):
        WLS.__init__(self, alpha_0, beta, sigma, tau, gamma, min_alpha, max_alpha)

    def search(self, problem: ExtendedProblem, x: np.array, f: np.array, d: np.array, theta: float, I: np.array = None):
        n = len(x)

        alpha = self._alpha_0
        alpha_l = 0
        alpha_u = np.inf

        first_iteration = True
        wolfe_cond_satisfied = False

        new_x = None
        new_f = None
        f_eval = 0

        if len(f.shape) > 1:
            f = f[np.where((f == problem.evaluate_functions(x)).all(axis=1))[0][0], :]
            f_eval += 1

        while not wolfe_cond_satisfied:

            if not first_iteration:
                alpha = (alpha_l + alpha_u) * self._delta if alpha_u is not np.inf else self._tau * max(alpha_l, self._alpha_0)
            else:
                first_iteration = False

            if alpha <= self._min_alpha or alpha >= self._max_alpha or alpha_u - alpha_l <= self._min_alpha:
                break

            new_x = x + alpha * d
            new_f = problem.evaluate_functions(new_x)
            f_eval += 1

            if not is_dominating_point(new_f, f, alpha, self._beta, theta) or np.isnan(new_f).any() or np.isinf(new_f).any():
                alpha_u = alpha
                continue
            else:
                if np.max(np.dot(problem.evaluate_functions_jacobian(new_x), d), axis=0) < self._sigma * theta:
                    f_eval += n

                    alpha_l = alpha
                    continue
            break

        if alpha <= self._min_alpha or alpha_u - alpha_l <= self._min_alpha:
            alpha = 0
            return None, None, alpha, f_eval
        elif alpha >= self._max_alpha:
            raise RuntimeError

        return new_x, new_f, alpha, f_eval
