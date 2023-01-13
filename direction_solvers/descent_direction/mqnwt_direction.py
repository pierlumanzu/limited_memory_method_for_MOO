import numpy as np
from gurobipy import Model, GRB, GurobiError

from nsma.direction_solvers.gurobi_settings import GurobiSettings

from direction_solvers.descent_direction.extended_dds import ExtendedDDS
from problems.extended_problem import ExtendedProblem


class MQNWTDirection(ExtendedDDS, GurobiSettings):

    def __init__(self, verbose: bool, gurobi_method: int, gurobi_verbose: bool):
        ExtendedDDS.__init__(self, verbose)
        GurobiSettings.__init__(self, gurobi_method, gurobi_verbose)

    def compute_direction(self, problem: ExtendedProblem, Jac: np.array, x_p: np.array = None, H: np.array = None):
        assert x_p is None and H is not None

        m, n = Jac.shape

        reset = False

        if np.isinf(Jac).any() or np.isnan(Jac).any() or np.isinf(H).any() or np.isnan(H).any():
            return np.zeros(n), 0, np.zeros(m), reset

        Jac_T = Jac.T

        eigvals = np.linalg.eigvals(H)
        min_eigval = np.min(eigvals)
        if min_eigval <= 0 and self._verbose:
            print('Warning Eigval: Min -> ', min_eigval, ' Max -> ', np.max(eigvals))
        if min_eigval < -5 * np.finfo(np.float).eps ** 0.5:
            if self._verbose:
                print('Reset')
            reset = True
            H = np.eye(n, dtype=np.float)

        try:
            model = Model("Modified Quasi Newton Direction -- Dual Problem")
            model.setParam("OutputFlag", self._gurobi_verbose)
            model.setParam("Method", self._gurobi_method)

            lam = model.addMVar(m, lb=0, ub=np.inf, name="lambda")

            obj = 1/2 * (lam @ np.dot(Jac, np.dot(H, Jac_T)) @ lam)
            model.setObjective(obj)

            linear_expression = sum(lam[i] for i in range(m))
            model.addConstr(linear_expression == 1, name='Lambda constraint')

            model.update()

            for i in range(m):
                lam[i].start = 0.

            model.optimize()

            if model.Status == GRB.OPTIMAL:
                sol = np.array([s.x for s in model.getVars()])
                d_p = -np.dot(np.dot(H, Jac_T), sol).flatten()
                theta_p = -model.getObjective().getValue()
            else:
                return np.zeros(n), 0, np.zeros(m), reset

        except GurobiError:
            if self._verbose:
                print('Gurobi Error')
            return np.zeros(n), 0, np.zeros(m), reset

        except AttributeError:
            return np.zeros(n), 0, np.zeros(m), reset

        return d_p, theta_p, sol, reset