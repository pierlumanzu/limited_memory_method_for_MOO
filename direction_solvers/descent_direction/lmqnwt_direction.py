import numpy as np
from gurobipy import Model, GRB, GurobiError

from nsma.direction_solvers.gurobi_settings import GurobiSettings

from direction_solvers.descent_direction.extended_dds import ExtendedDDS
from problems.extended_problem import ExtendedProblem


class LMQNWTDirection(ExtendedDDS, GurobiSettings):

    def __init__(self, verbose: bool, gurobi_method: int, gurobi_verbose: bool):
        ExtendedDDS.__init__(self, verbose)
        GurobiSettings.__init__(self, gurobi_method, gurobi_verbose)

    def compute_direction(self, problem: ExtendedProblem, Jac: np.array, x_p: np.array = None, HG_results: np.array = None):
        assert x_p is None and HG_results is not None

        m, n = Jac.shape

        P = np.dot(Jac, HG_results)
        reset = False

        if np.isinf(P).any() or np.isnan(P).any():
            return np.zeros(n), 0, np.zeros(m), reset

        eigvals = np.linalg.eigvals(P)
        min_eigval = np.min(eigvals)
        if min_eigval <= 0 and self._verbose:
            print('Warning Eigval: Min -> ', min_eigval, ' Max -> ', np.max(eigvals))
        if min_eigval < -5 * np.finfo(np.float).eps ** 0.5:
            if self._verbose:
                print('Reset')
            reset = True
            P = np.dot(Jac, Jac.T)

        try:
            model = Model("Modified L-BFGS Direction -- Dual Problem")
            model.setParam("OutputFlag", self._gurobi_verbose)
            model.setParam("Method", self._gurobi_method)

            lam = model.addMVar(m, lb=0, ub=np.inf, name="lambda")

            obj = 1/2 * (lam @ P @ lam)
            model.setObjective(obj)

            linear_expression = sum(lam[i] for i in range(m))
            model.addConstr(linear_expression == 1, name='Lambda constraint')

            model.update()

            for i in range(m):
                lam[i].start = 0.

            model.optimize()

            if model.Status == GRB.OPTIMAL:
                sol = np.array([s.x for s in model.getVars()])
                d_p = -np.dot(HG_results, sol).flatten()
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
