import numpy as np
from datetime import datetime
import tensorflow as tf

from nsma.algorithms.algorithm_utils.graphical_plot import GraphicalPlot
from nsma.general_utils.pareto_utils import points_initialization, points_postprocessing

from algorithms.algorithm_factory import AlgorithmFactory
from general_utils.args_utils import print_parameters, args_preprocessing, args_file_creation
from general_utils.management_utils import folder_initialization, execution_time_file_initialization, write_in_execution_time_file, write_results_in_csv_file_front_mode, write_results_in_csv_file_local_search_mode, save_plots
from general_utils.progress_bar import ProgressBarWrapper
from constants import PROBLEM_DIMENSIONS
from parser_management import get_args

tf.compat.v1.disable_eager_execution()

args = get_args()

print_parameters(args)
algorithms_names, problems, n_problems, front_mode, general_settings, algorithms_settings, DDS_settings, ALS_settings, WLS_settings = args_preprocessing(args)
print('N° algorithms: ', len(algorithms_names))
print('N° problems: ', n_problems)
if front_mode:
    print('N° seeds', len(general_settings['seeds']))
else:
    print('N° num trials: ', len(general_settings['num_trials']))
print()

date = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

if general_settings['verbose']:
    progress_bar = ProgressBarWrapper(len(algorithms_names) * n_problems * len(general_settings['seeds' if front_mode else 'num_trials']))
    progress_bar.show_bar()

cycle_items = general_settings['seeds'] if front_mode else general_settings['num_trials']

for cycle_item in cycle_items:
    print()
    if front_mode:
        print('Seed ', cycle_item)
    else:
        print('N° trials', cycle_item)

    if general_settings['general_export']:
        folder_initialization(cycle_item, date, algorithms_names, front_mode)
        args_file_creation(cycle_item, date, args)
        if front_mode:
            execution_time_file_initialization(cycle_item, date, algorithms_names)

    for problem in problems:
        print()
        print('Problem: ', problem.name())

        var_range = PROBLEM_DIMENSIONS[problem.family_name()]

        for n in var_range:
            print()
            print()
            print('N: ', n)

            for index_algorithm, algorithm_name in enumerate(algorithms_names):
                print()
                print('Algorithm: ', algorithm_name)

                session = tf.compat.v1.Session()
                with session.as_default():

                    problem_instance = problem(n=n)

                    if front_mode:
                        np.random.seed(cycle_item)
                    else:
                        assert len(general_settings['seeds']) == 1
                        np.random.seed(general_settings['seeds'][0])

                    if not index_algorithm:
                        if front_mode:
                            assert len(general_settings['num_trials']) == 1
                            initial_p_list, initial_f_list, n_initial_points = points_initialization(problem_instance, 'rand', general_settings['num_trials'][0])
                        else:
                            initial_p_list, initial_f_list, n_initial_points = points_initialization(problem_instance, 'rand', cycle_item)

                    algorithm = AlgorithmFactory.get_algorithm(algorithm_name,
                                                               general_settings=general_settings,
                                                               algorithms_settings=algorithms_settings,
                                                               DDS_settings=DDS_settings,
                                                               ALS_settings=ALS_settings,
                                                               WLS_settings=WLS_settings)

                    if front_mode:
                        p_list, f_list, elapsed_time = algorithm.search(np.copy(initial_p_list), np.copy(initial_f_list), problem_instance)
                        p_list, f_list = points_postprocessing(np.copy(p_list), np.copy(f_list), problem_instance)
                    else:
                        p_list, f_list, elapsed_time, n_iterations, n_f_evals, n_g_evals, n_h_evals, thetas = algorithm.search(np.copy(initial_p_list), np.copy(initial_f_list), problem_instance)

                    if general_settings['plot_pareto_front']:
                        graphical_plot = GraphicalPlot(general_settings['plot_pareto_solutions'], general_settings['plot_dpi'])
                        graphical_plot.show_figure(p_list, f_list, hold_still=True)
                        graphical_plot.close_figure()

                    if general_settings['general_export']:
                        if front_mode:
                            write_in_execution_time_file(cycle_item, date, algorithm_name, problem, n, elapsed_time)
                            write_results_in_csv_file_front_mode(p_list, f_list, cycle_item, date, algorithm_name, problem, export_pareto_solutions=general_settings['export_pareto_solutions'])
                        else:
                            write_results_in_csv_file_local_search_mode(p_list, f_list, elapsed_time, n_iterations, n_f_evals, n_g_evals, n_h_evals, thetas, cycle_item, date, algorithm_name, problem, export_pareto_solutions=general_settings['export_pareto_solutions'])
                        try:
                            save_plots(p_list, f_list, cycle_item, date, algorithm_name, problem, general_settings['export_pareto_solutions'], general_settings['plot_dpi'])
                        except OverflowError:
                            not_overflow_inducing_indices = np.where(np.all(f_list <= 1e300, axis=1))[0]
                            p_list = p_list[not_overflow_inducing_indices, :]
                            f_list = f_list[not_overflow_inducing_indices, :]
                            save_plots(p_list, f_list, cycle_item, date, algorithm_name, problem, general_settings['export_pareto_solutions'], general_settings['plot_dpi'])

                    if general_settings['verbose']:
                        progress_bar.increment_current_value()
                        progress_bar.show_bar()

                    tf.compat.v1.reset_default_graph()
                    session.close()
