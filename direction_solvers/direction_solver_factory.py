from direction_solvers.descent_direction.steepest_direction import SteepestDirection
from direction_solvers.descent_direction.nwt_direction import NWTDirection
from direction_solvers.descent_direction.qnwt_direction import QNWTDirection
from direction_solvers.descent_direction.mqnwt_direction import MQNWTDirection
from direction_solvers.descent_direction.lmqnwt_direction import LMQNWTDirection


class DirectionSolverFactory:

    @staticmethod
    def get_direction_calculator(direction_type: str, verbose: bool, gurobi_method: int, gurobi_verbose: bool):

        if direction_type == 'SteepestDirection':
            return SteepestDirection(verbose, gurobi_method, gurobi_verbose)

        elif direction_type == 'NWTDirection':
            return NWTDirection(verbose, gurobi_method, gurobi_verbose)

        elif direction_type == 'QNWTDirection':
            return QNWTDirection(verbose, gurobi_method, gurobi_verbose)

        elif direction_type == 'MQNWTDirection':
            return MQNWTDirection(verbose, gurobi_method, gurobi_verbose)

        elif direction_type == 'LMQNWTDirection':
            return LMQNWTDirection(verbose, gurobi_method, gurobi_verbose)

        else:
            raise NotImplementedError
