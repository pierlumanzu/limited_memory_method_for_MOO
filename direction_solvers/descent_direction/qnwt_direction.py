import numpy as np
from gurobipy import Model, GRB, GurobiError

from nsma.direction_solvers.gurobi_settings import GurobiSettings

from direction_solvers.descent_direction.extended_dds import ExtendedDDS
from problems.extended_problem import ExtendedProblem


class QNWTDirection(ExtendedDDS, GurobiSettings):

    def __init__(self, verbose: bool, gurobi_method: int, gurobi_verbose: bool):
        ExtendedDDS.__init__(self, verbose)
        GurobiSettings.__init__(self, gurobi_method, gurobi_verbose)

    def compute_direction(self, problem: ExtendedProblem, Jac: np.array, x_p: np.array = None, B: np.array = None):
        assert x_p is None and B is not None

        m, n = Jac.shape

        reset = False

        if np.isinf(Jac).any() or np.isnan(Jac).any() or np.isinf(B).any() or np.isnan(B).any():
            return np.zeros(n), 0, reset

        for j in range(m):
            eigvals = np.linalg.eigvals(B[j, :, :])
            min_eigval = np.min(eigvals)
            if min_eigval <= 0 and self._verbose:
                print(j, ' Warning Eigval: Min -> ', min_eigval, ' Max -> ', np.max(eigvals))
            if min_eigval < -5 * np.finfo(np.float).eps ** 0.5:
                reset = True
        if reset:
            if self._verbose:
                print('Reset')
            B = np.empty((m, n, n), dtype=np.float)
            for j in range(m):
                B[j, :, :] = np.eye(n, dtype=np.float)

        try:
            model = Model("Quasi Newton Direction")
            model.setParam("OutputFlag", self._gurobi_verbose)
            model.setParam("Method", self._gurobi_method)

            d = model.addMVar(n, lb=-np.inf, ub=np.inf, name="d")
            t = model.addMVar(1, lb=-np.inf, ub=0., name="t")

            obj = t
            model.setObjective(obj)

            for j in range(m):
                linear_expression = Jac[j, :] @ d + 1/2 * (d @ B[j, :, :] @ d) - t
                model.addConstr(linear_expression <= 0, name='Jacobian/B constraint n°{}'.format(j))

            model.update()

            for i in range(n):
                d[i].start = 0.
            t.start = 0.

            model.optimize()

            if model.Status == GRB.OPTIMAL:
                sol = model.getVars()
                d_p = np.array([s.x for s in sol][:n])
                theta_p = sol[-1].x
            else:
                return np.zeros(n), 0, reset

        except GurobiError:
            if self._verbose:
                print('Gurobi Error')
            return np.zeros(n), 0, reset

        except AttributeError:
            return np.zeros(n), 0, reset

        return d_p, theta_p, reset