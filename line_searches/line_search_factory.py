from line_searches.wolfe_type.mo_wls import MOWLS


class LineSearchFactory:

    @staticmethod
    def get_line_search(line_search_name: str, args_line_search: dict):

        if line_search_name == 'MOWLS':
            return MOWLS(args_line_search['WLS_alpha_0'],
                         args_line_search['WLS_beta'],
                         args_line_search['WLS_sigma'],
                         args_line_search['WLS_tau'],
                         args_line_search['WLS_gamma'],
                         args_line_search['WLS_min_alpha'],
                         args_line_search['WLS_max_alpha'])

        else:
            raise NotImplementedError
