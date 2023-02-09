from nsma.algorithms.gradient_based.local_search_algorithms.fmopg import FMOPG

from direction_solvers.direction_solver_factory import DirectionSolverFactory


class ArmijoBasedFMOPG(FMOPG):

    def __init__(self,
                 theta_tol: float,
                 gurobi_method: int,
                 gurobi_verbose: bool,
                 args_line_search: dict,
                 max_iter: int = None,
                 max_time: float = None,
                 max_f_evals: int = None):

        FMOPG.__init__(self,
                       theta_tol,
                       True,
                       gurobi_method,
                       gurobi_verbose,
                       args_line_search['ALS_alpha_0'], args_line_search['ALS_delta'], args_line_search['ALS_beta'], args_line_search['ALS_min_alpha'],
                       max_iter,
                       max_time,
                       max_f_evals)

        self._direction_solver = DirectionSolverFactory.get_direction_calculator('SteepestDirectionForFMOPG', False, gurobi_method, gurobi_verbose)