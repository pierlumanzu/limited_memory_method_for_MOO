import numpy as np


def is_dominating_point(new_f: np.array, f: np.array, alpha: float, beta: float, theta: float):

    if np.isnan(new_f).any():
        return False

    n_obj = len(new_f)

    suff_decr = np.reshape(f + beta * alpha * theta, (1, n_obj))
    dominance_matrix = new_f - suff_decr
    is_dominating = (np.sum(dominance_matrix <= 0, axis=1) == n_obj).any()

    return is_dominating