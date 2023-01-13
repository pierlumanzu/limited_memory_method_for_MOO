import os
import numpy as np

from nsma.algorithms.algorithm_utils.graphical_plot import GraphicalPlot

from problems.extended_problem import ExtendedProblem


def make_folder(folder_path: str):
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)


def folder_initialization(cycle_item: int, date: str, algorithms_names: list, front_mode: bool):
    assert os.path.exists(os.path.join('Execution_Outputs'))

    folders = ['Execution_Times', 'Csv', 'Plot'] if front_mode else ['Csv', 'Plot']

    path = os.path.join('Execution_Outputs', date)
    make_folder(path)

    path = os.path.join(path, str(cycle_item))
    make_folder(path)

    for index_folder, folder in enumerate(folders):
        make_folder(os.path.join(path, folder))
        if index_folder >= (1 if front_mode else 0):
            for algorithm_name in algorithms_names:
                make_folder(os.path.join(path, folder, algorithm_name))


def execution_time_file_initialization(seed: int, date: str, algorithms_names: list):
    for algorithm_name in algorithms_names:
        execution_time_file = open(os.path.join('Execution_Outputs', date, str(seed), 'Execution_Times', '{}.txt'.format(algorithm_name)), 'w')
        execution_time_file.close()

def write_in_execution_time_file(seed: int, date: str, algorithm_name: str, problem: ExtendedProblem, n: int, elapsed_time: float):
    execution_time_file = open(os.path.join('Execution_Outputs', date, str(seed), 'Execution_Times', '{}.txt'.format(algorithm_name)), 'a')
    execution_time_file.write('Problem: ' + problem.__name__ + '    N: ' + str(n) + '    Time: ' + str(elapsed_time) + '\n')
    execution_time_file.close()

def write_results_in_csv_file_front_mode(p_list: np.array, f_list: np.array, seed: int, date: str, algorithm_name: str, problem: ExtendedProblem, export_pareto_solutions: bool = False):
    assert len(p_list) == len(f_list)
    n = p_list.shape[1]

    f_list_file = open(os.path.join('Execution_Outputs', date, str(seed), 'Csv', algorithm_name, '{}_{}_pareto_front.csv'.format(problem.__name__, n)), 'w')
    if len(f_list):
        for i in range(f_list.shape[0]):
            f_list_file.write(';'.join([str(el) for el in f_list[i, :]]) + '\n')
    f_list_file.close()

    if export_pareto_solutions:
        p_list_file = open(os.path.join('Execution_Outputs', date, str(seed), 'Csv', algorithm_name, '{}_{}_pareto_solutions.csv'.format(problem.__name__, n)), 'w')
        if len(p_list):
            for i in range(p_list.shape[0]):
                p_list_file.write(';'.join([str(el) for el in p_list[i, :]]) + '\n')
        p_list_file.close()

def write_results_in_csv_file_local_search_mode(p_list: np.array, f_list: np.array, elapsed_time: np.array, n_iterations: np.array, n_f_evals: np.array, n_g_evals: np.array, n_h_evals: np.array, thetas: np.array, num_trials: int, date: str, algorithm_name: str, problem: ExtendedProblem, export_pareto_solutions: bool = False):
    write_results_in_csv_file_front_mode(p_list, f_list, num_trials, date, algorithm_name, problem, export_pareto_solutions)
    n = p_list.shape[1]

    metrics_file = open(os.path.join('Execution_Outputs', date, str(num_trials), 'Csv', algorithm_name, '{}_{}_metrics.csv'.format(problem.__name__, n)), 'w')
    if len(elapsed_time):
        assert len(elapsed_time) == len(n_iterations) == len(n_f_evals) == len(n_g_evals) == len(n_h_evals)
        metrics_file.write(';'.join(['N° Point', 'Elapsed Time (s)', 'N° Iterations', 'N° Functions Evaluation', 'N° Jacobian Evaluation', 'N° Hessian Evaluation', 'Theta (Steepest Descent)']) + '\n')
        for i in range(len(elapsed_time)):
            metrics_file.write(';'.join([str(i + 1), str(round(elapsed_time[i], 10)).replace('.', ','), str(n_iterations[i]), str(n_f_evals[i]), str(n_g_evals[i]), str(n_h_evals[i]), str(round(thetas[i], 10)).replace('.', ',')]) + '\n')
    metrics_file.close()


def save_plots(p_list: np.array, f_list: np.array, cycle_item: int, date: str, algorithm_name: str, problem: ExtendedProblem, export_pareto_solutions: bool, plot_dpi: int):
    assert len(p_list) == len(f_list)

    graphical_plot = GraphicalPlot(export_pareto_solutions, plot_dpi)
    graphical_plot.save_figure(p_list, f_list, os.path.join('Execution_Outputs', date, str(cycle_item), 'Plot'), algorithm_name, problem.__name__)
