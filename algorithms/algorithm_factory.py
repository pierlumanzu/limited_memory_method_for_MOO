from algorithms.gradient_based.nwt import NWT
from algorithms.gradient_based.q_nwt import QNWT
from algorithms.gradient_based.mq_nwt import MQNWT
from algorithms.gradient_based.lm_q_nwt import LMQNWT

from algorithms.memetic.armijo_based_nsma import ArmijoBasedNSMA
from algorithms.memetic.wolfe_based_nsma import WolfeBasedNSMA
from algorithms.memetic.lm_q_nwt_based_nsma import LMQNWTBasedNSMA


class AlgorithmFactory:

    @staticmethod
    def get_algorithm(algorithm_name, **kwargs):

        general_settings = kwargs['general_settings']

        algorithms_settings = kwargs['algorithms_settings']

        if algorithm_name == 'NWT':
            NWT_settings = algorithms_settings[algorithm_name]

            DDS_settings = kwargs['DDS_settings']
            WLS_settings = kwargs['WLS_settings']

            algorithm = NWT(general_settings['max_iter'],
                            general_settings['max_time'],
                            general_settings['max_f_evals'],
                            general_settings['verbose'],
                            general_settings['verbose_interspace'],
                            general_settings['plot_pareto_front'],
                            general_settings['plot_pareto_solutions'],
                            general_settings['plot_dpi'],
                            NWT_settings['theta_tol'],
                            DDS_settings['gurobi_method'],
                            DDS_settings['gurobi_verbose'],
                            WLS_settings)

        elif algorithm_name == 'QNWT':
            QNWT_settings = algorithms_settings[algorithm_name]

            DDS_settings = kwargs['DDS_settings']
            WLS_settings = kwargs['WLS_settings']

            algorithm = QNWT(general_settings['max_iter'],
                             general_settings['max_time'],
                             general_settings['max_f_evals'],
                             general_settings['verbose'],
                             general_settings['verbose_interspace'],
                             general_settings['plot_pareto_front'],
                             general_settings['plot_pareto_solutions'],
                             general_settings['plot_dpi'],
                             QNWT_settings['theta_tol'],
                             DDS_settings['gurobi_method'],
                             DDS_settings['gurobi_verbose'],
                             WLS_settings)

        elif algorithm_name == 'MQNWT':
            MQNWT_settings = algorithms_settings[algorithm_name]

            DDS_settings = kwargs['DDS_settings']
            WLS_settings = kwargs['WLS_settings']

            algorithm = MQNWT(general_settings['max_iter'],
                              general_settings['max_time'],
                              general_settings['max_f_evals'],
                              general_settings['verbose'],
                              general_settings['verbose_interspace'],
                              general_settings['plot_pareto_front'],
                              general_settings['plot_pareto_solutions'],
                              general_settings['plot_dpi'],
                              MQNWT_settings['theta_tol'],
                              DDS_settings['gurobi_method'],
                              DDS_settings['gurobi_verbose'],
                              WLS_settings)

        elif algorithm_name == 'LMQNWT':
            LMQNWT_settings = algorithms_settings[algorithm_name]

            DDS_settings = kwargs['DDS_settings']
            WLS_settings = kwargs['WLS_settings']

            algorithm = LMQNWT(general_settings['max_iter'],
                               general_settings['max_time'],
                               general_settings['max_f_evals'],
                               general_settings['verbose'],
                               general_settings['verbose_interspace'],
                               general_settings['plot_pareto_front'],
                               general_settings['plot_pareto_solutions'],
                               general_settings['plot_dpi'],
                               LMQNWT_settings['theta_tol'],
                               LMQNWT_settings['max_cor'],
                               DDS_settings['gurobi_method'],
                               DDS_settings['gurobi_verbose'],
                               WLS_settings)

        elif algorithm_name == 'ArmijoBasedNSMA':
            ArmijoBasedNSMA_settings = algorithms_settings[algorithm_name]

            DDS_settings = kwargs['DDS_settings']
            ALS_settings = kwargs['ALS_settings']

            algorithm = ArmijoBasedNSMA(general_settings['max_iter'],
                                        general_settings['max_time'],
                                        general_settings['max_f_evals'],
                                        general_settings['verbose'],
                                        general_settings['verbose_interspace'],
                                        general_settings['plot_pareto_front'],
                                        general_settings['plot_pareto_solutions'],
                                        general_settings['plot_dpi'],
                                        ArmijoBasedNSMA_settings['pop_size'],
                                        ArmijoBasedNSMA_settings['crossover_probability'],
                                        ArmijoBasedNSMA_settings['crossover_eta'],
                                        ArmijoBasedNSMA_settings['mutation_eta'],
                                        ArmijoBasedNSMA_settings['shift'],
                                        ArmijoBasedNSMA_settings['crowding_quantile'],
                                        ArmijoBasedNSMA_settings['n_opt'],
                                        ArmijoBasedNSMA_settings['FMOPG_max_iter'],
                                        ArmijoBasedNSMA_settings['theta_for_stationarity'],
                                        ArmijoBasedNSMA_settings['theta_tol'],
                                        ArmijoBasedNSMA_settings['theta_dec_factor'],
                                        DDS_settings['gurobi_method'],
                                        DDS_settings['gurobi_verbose'],
                                        ALS_settings)

        elif algorithm_name == 'WolfeBasedNSMA':
            WolfeBasedNSMA_settings = algorithms_settings[algorithm_name]

            DDS_settings = kwargs['DDS_settings']
            WLS_settings = kwargs['WLS_settings']

            algorithm = WolfeBasedNSMA(general_settings['max_iter'],
                                       general_settings['max_time'],
                                       general_settings['max_f_evals'],
                                       general_settings['verbose'],
                                       general_settings['verbose_interspace'],
                                       general_settings['plot_pareto_front'],
                                       general_settings['plot_pareto_solutions'],
                                       general_settings['plot_dpi'],
                                       WolfeBasedNSMA_settings['pop_size'],
                                       WolfeBasedNSMA_settings['crossover_probability'],
                                       WolfeBasedNSMA_settings['crossover_eta'],
                                       WolfeBasedNSMA_settings['mutation_eta'],
                                       WolfeBasedNSMA_settings['shift'],
                                       WolfeBasedNSMA_settings['crowding_quantile'],
                                       WolfeBasedNSMA_settings['n_opt'],
                                       WolfeBasedNSMA_settings['FMOPG_max_iter'],
                                       WolfeBasedNSMA_settings['theta_for_stationarity'],
                                       WolfeBasedNSMA_settings['theta_tol'],
                                       WolfeBasedNSMA_settings['theta_dec_factor'],
                                       DDS_settings['gurobi_method'],
                                       DDS_settings['gurobi_verbose'],
                                       WLS_settings)

        elif algorithm_name == 'LMQNWTBasedNSMA':
            LMQNWTBasedNSMA_settings = algorithms_settings[algorithm_name]

            DDS_settings = kwargs['DDS_settings']
            WLS_settings = kwargs['WLS_settings']

            algorithm = LMQNWTBasedNSMA(general_settings['max_iter'],
                                        general_settings['max_time'],
                                        general_settings['max_f_evals'],
                                        general_settings['verbose'],
                                        general_settings['verbose_interspace'],
                                        general_settings['plot_pareto_front'],
                                        general_settings['plot_pareto_solutions'],
                                        general_settings['plot_dpi'],
                                        LMQNWTBasedNSMA_settings['pop_size'],
                                        LMQNWTBasedNSMA_settings['crossover_probability'],
                                        LMQNWTBasedNSMA_settings['crossover_eta'],
                                        LMQNWTBasedNSMA_settings['mutation_eta'],
                                        LMQNWTBasedNSMA_settings['shift'],
                                        LMQNWTBasedNSMA_settings['crowding_quantile'],
                                        LMQNWTBasedNSMA_settings['n_opt'],
                                        LMQNWTBasedNSMA_settings['LMQNWT_max_iter'],
                                        LMQNWTBasedNSMA_settings['LMQNWT_max_cor'],
                                        LMQNWTBasedNSMA_settings['theta_for_stationarity'],
                                        LMQNWTBasedNSMA_settings['theta_tol'],
                                        LMQNWTBasedNSMA_settings['theta_dec_factor'],
                                        DDS_settings['gurobi_method'],
                                        DDS_settings['gurobi_verbose'],
                                        WLS_settings)

        else:
            raise NotImplementedError

        return algorithm
