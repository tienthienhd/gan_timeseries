import GPyOpt
import run
import numpy as np
import os
import multiprocessing as mp
from GPyOpt.methods import BayesianOptimization

from data import DataSets


def fitness_single(param):

    dataset = DataSets('../data/gg_trace/5.csv',
                       usecols=[3],
                       column_names=['cpu'],
                       header=None,
                       n_in=int(param[0]),
                       n_out=int(param[1]),
                       is_diff=True,
                       is_log=True,
                       is_stand=True,
                       feature_range=(-1, 1))
    config_gru_gan = {
        "params_generator": {
            "layer_size": [int(param[2]), dataset.get_input_shape()[-1]],
            "activation": 'tanh',
            "dropout": param[3],
            "output_activation": "tanh",
            "cell_type": "gru",
            "concat_noise": "after"

        },
        "params_discriminator": {
            "layer_size": [int(param[4]), dataset.get_input_shape()[-1]],
            "activation": 'tanh',
            "dropout": param[5],
            "output_activation": 'sigmoid',
            "cell_type": "gru"

        },
        "input_shape": dataset.get_input_shape(),
        "noise_shape": dataset.get_input_shape(),
        "optimizer_g": 'rmsprop',
        "optimizer_d": 'rmsprop',
        "learning_rate_g": param[6],
        "learning_rate_d": param[6],
        "num_train_d": int(param[7]),
        "is_wgan": False,
        "model_dir": "logs/ann_gan"
    }

    config_train = {
        "validation_split": 0.2,
        "batch_size": int(param[8]),
        # "batch_size": 2000,
        "epochs": 200,
        "verbose": 0,
        "step_print": 1
    }
    return run.run('GruGan', config_init=config_gru_gan, config_train=config_train, dataset=dataset)


def fitness(params):
    res = fitness_single(params[0])
    return res


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

    opt = BayesianOptimization(f=fitness,
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
