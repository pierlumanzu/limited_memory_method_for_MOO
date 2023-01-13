from abc import ABC

from nsma.direction_solvers.descent_direction.dds import DDS


class ExtendedDDS(DDS, ABC):

    def __init__(self, verbose: bool):
        DDS.__init__(self)
        self._verbose = verbose