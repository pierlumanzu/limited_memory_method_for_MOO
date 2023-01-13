from problems.jos.jos_instances import JOS1a, JOS1b, JOS1c
from problems.mop.mop_instances import MMOP2, MOP3, MOP7
from problems.slc.slc_instance import SLC2
from problems.mmr.mmr_instance import MMR5
from problems.cec.cec09_instances import CEC091, CEC092, CEC093, CEC097, CEC098, CEC0910
from problems.man.man_instances import MMAN1, MAN2
from problems.m_fds.m_fds_instance import MFDS1


PROBLEMS = {
    'JOS': [JOS1a],
    'MOP': [MMOP2, MOP3],
    'MOP7': [MOP7],
    'SLC': [SLC2],
    'MMR': [MMR5],
    'CEC': [CEC091, CEC092, CEC093, CEC097, CEC098, CEC0910],
    'MAN': [MAN2],
    'MFDS': [MFDS1]
}


PROBLEM_DIMENSIONS = {
    'JOS': [2],
    'MMOP2': [2, 5, 10, 20, 30, 40, 50, 100, 200, 500, 1000],
    'MOP_3_7': [2],
    'SLC2': [2, 5, 10, 20, 30, 40, 50, 100, 200, 500, 1000],
    'MMR': [2, 5, 10, 20, 30, 40, 50, 100, 200, 500, 1000],
    'CEC': [5, 10, 20, 30, 40, 50, 100, 200, 500, 1000],
    'MAN': [30],
    'MFDS': [2, 5, 10, 20, 30, 40, 50, 100, 200, 500, 1000]
}
