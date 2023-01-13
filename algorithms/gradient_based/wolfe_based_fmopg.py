from nsma.algorithms.gradient_based.local_search_algorithms.fmopg import FMOPG

from line_searches.line_search_factory import LineSearchFactory

class WolfeBasedFMOPG(FMOPG):

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
                       0., 0., 0., 0.,
                       max_iter,
                       max_time,
                       max_f_evals)

        self._line_search = LineSearchFactory.get_line_search('MOWLS', args_line_search)