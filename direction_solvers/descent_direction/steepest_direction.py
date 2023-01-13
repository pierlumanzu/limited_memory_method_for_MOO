import numpy as np
from gurobipy import Model, GRB, GurobiError

from nsma.direction_solvers.gurobi_settings import GurobiSettings

from direction_solvers.descent_direction.extended_dds import ExtendedDDS
from problems.extended_problem import ExtendedProblem


class SteepestDirection(ExtendedDDS, GurobiSettings):

    def __init__(self, verbose: bool, gurobi_method: int, gurobi_verbose: bool):
        ExtendedDDS.__init__(self, verbose)
        GurobiSettings.__init__(self, gurobi_method, gurobi_verbose)

    def compute_direction(self, problem: ExtendedProblem, Jac: np.array, x_p: np.array = None):
        assert x_p is None

        m, n = Jac.shape

        P = np.dot(Jac, Jac.T)

        if np.isinf(Jac).any() or np.isnan(Jac).any():
            return np.zeros(n), 0, np.zeros(m)

        try:
            model = Model("Steepest Direction -- Dual Problem")
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
                d_p = -np.dot(Jac.T, sol).flatten()
                theta_p = -model.getObjective().getValue()
            else:
                return np.zeros(n), 0, np.zeros(m)

        except GurobiError:
            if self._verbose:
                print('Gurobi Error')
            return np.zeros(n), 0, np.zeros(m)

        except AttributeError:
            return np.zeros(n), 0, np.zeros(m)

        return d_p, theta_p, sol
