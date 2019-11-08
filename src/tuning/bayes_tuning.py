import GPyOpt
import run
import numpy as np
import os
import multiprocessing as mp
from GPyOpt.methods import BayesianOptimization
from tuning.function import custom_fitness


if __name__ == '__main__':
    # layer_sizes = [[4], [8], [16], [32], [64], [8, 4], [16, 8], [16, 4], [32, 16], [32, 8], [64, 16], [16, 8, 4],
    #                [32, 16, 8], [32, 16, 4]]
    # activations = ['tanh', 'sigmoid', 'relu']
    # n_train_ds = [1, 2, 3, 4]

    model_tuning = "gru_gan"
    path_log = "../logs/tuning/" + model_tuning

    domain = [
        {'name': 'n_in', 'type': 'discrete', 'domain': [1, 2, 3, 4, 5, 6, 7, 8]},
        {'name': 'n_out', 'type': 'discrete', 'domain': [1]},
        {'name': 'g_layer_size', 'type': 'discrete', 'domain': [2, 4, 8, 16, 32, 64]},
        {'name': 'g_dropout', 'type': 'continuous', 'domain': (0, 0.8)},

        {'name': 'd_layer_size', 'type': 'discrete', 'domain': [2, 4, 8, 16, 32, 64]},
        {'name': 'd_dropout', 'type': 'continuous', 'domain': (0, 0.8)},
        {'name': 'learning_rate', 'type': 'continuous', 'domain': (0.0001, 0.1)},
        {'name': 'num_train_d', 'type': 'discrete', 'domain': [1, 2, 3, 4, 5]},
        {'name': 'batch_size', 'type': 'discrete', 'domain': [4, 8, 16, 32]},


    ]
    constraints = []

    opt = BayesianOptimization(f=custom_fitness,
                               domain=domain,
                               constraints=constraints,
                               num_cores=6,
                               batch_size=4,
                               initial_design_numdata=20)

    opt.run_optimization(max_iter=100, max_time=np.inf, verbosity=True,
                         report_file=os.path.join(path_log, 'report.txt'),
                         evaluations_file=os.path.join(path_log, 'evaluations.txt'),
                         models_file=os.path.join(path_log, 'model_file.txt'))
    opt.plot_convergence()
    # print(opt.X)
    print(opt.Y)
