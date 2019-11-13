import sys

sys.path.append("src/")
#import run
from data import DataSets
from tuning import function
import pandas as pd
import numpy as np
from metrics import evaluate
import glob


def run_tuning():
    import src.tuning.bayes_tuning
    # import src.tuning.metaheuristic.pso_lib


def process_result():
    result = []
    for file in glob.glob("result/tuning/bayes/*/*/*/*/*.csv"):
        res = get_info(file)
        result.append(res)
    df = pd.DataFrame(result)
    df.to_csv("result/tuning/bayes/result_tuning.csv", index=False)


def get_info(file_path):
    tmp = file_path.split("/")
    filename = tmp[-1]
    sub_data = tmp[-2]
    data = tmp[-3]
    loss_type = tmp[-4]
    algo = tmp[-5]
    tuning_algo = tmp[-6]
    df = pd.read_csv(file_path)

    actual = df.iloc[:, 0].values

    predicts = df.iloc[:, 1:].values
    predict_mean = np.mean(predicts, axis=1)
    predict_std = np.std(predicts, axis=1)
    mean_std = np.mean(predict_std)
    result = {
        "filename": filename,
        "data": data,
        "sub_data": sub_data,
        "algo": algo,
        "loss_type": loss_type,
        "tuning_algo": tuning_algo,
        "mean_std": mean_std
    }
    res_eval = evaluate(actual, predict_mean, metrics=('mae', 'rmse', 'mape', 'smape', "std_ae", 'std_ape', "jsd"))
    result.update(res_eval)
    return result


def test():
    dataset = DataSets('data/gg_trace/5.csv',
                       usecols=[3],
                       column_names=['cpu'],
                       header=None,
                       n_in=1,
                       n_out=1,
                       is_diff=True,
                       is_log=True,
                       is_stand=True,
                       test_size=0.2,
                       feature_range=(-1, 1))

    config_ann = {
        "params": {
            "layer_size": [32, dataset.get_input_shape()[-1]],
            "activation": 'tanh',
            "dropout": 0.5,
            "output_activation": None

        },
        "input_shape": dataset.get_input_shape(),
        "optimizer": 'adam',
        "learning_rate": 0.001,
        "model_dir": "test"
    }

    config_flnn = {
        "params": {
            "list_function": ['sin', 'cos', 'tan'],
            "activation": 'tanh',
            "num_output": 1

        },
        "input_shape": dataset.get_input_shape(),
        "optimizer": 'adam',
        "learning_rate": 0.001,
        "model_dir": "test"

    }

    config_flnn_gan = {
        "params_generator": {
            "list_function": ['sin', 'cos', 'tan'],
            "activation": 'tanh',
            "num_output": 1

        },
        "params_discriminator": {
            "list_function": ['sin', 'cos', 'tan'],
            "activation": 'tanh',
            "num_output": 1

        },
        "input_shape": dataset.get_input_shape(),
        "noise_shape": [1, 1],
        "optimizer_g": 'adam',
        "optimizer_d": 'adam',
        "learning_rate_g": 0.001,
        "learning_rate_d": 0.001,
        "num_train_d": 1,
        "loss_type": "loss_gan",
        "is_wgan": False,
        "model_dir": "logs/flnn_gan"
    }

    config_ann_gan = {
        "params_generator": {
            "layer_size": [8, dataset.get_input_shape()[-1]],
            "activation": 'sigmoid',
            "dropout": 0,
            "output_activation": None

        },
        "params_discriminator": {
            "layer_size": [16, dataset.get_input_shape()[-1]],
            "activation": 'sigmoid',
            "dropout": 0,
            "output_activation": 'sigmoid'

        },
        "input_shape": dataset.get_input_shape(),
        "noise_shape": dataset.get_input_shape(),
        "optimizer_g": 'rmsprop',
        "optimizer_d": 'rmsprop',
        "learning_rate_g": 0.001,
        "learning_rate_d": 0.001,
        "num_train_d": 2,
        "loss_type": "loss_gan",
        "is_wgan": False,
        "model_dir": "logs/ann_gan"
    }

    config_gru_gan = {
        "params_generator": {
            "layer_size": [2, dataset.get_input_shape()[-1]],
            "activation": 'tanh',
            "dropout": 0.273,
            "output_activation": "tanh",
            "cell_type": "gru",
            "concat_noise": "after"
        },
        "params_discriminator": {
            "layer_size": [16, dataset.get_input_shape()[-1]],
            "activation": 'tanh',
            "dropout": 0.308,
            "output_activation": 'sigmoid',
            "cell_type": "gru"

        },
        "input_shape": dataset.get_input_shape(),
        "output_shape": dataset.get_output_shape(),
        "noise_shape": [32, 1],
        "optimizer_g": 'adam',
        "optimizer_d": 'adam',
        "learning_rate_g": 0.004,
        "learning_rate_d": 0.01,
        "num_train_d": 5,
        "loss_type": "loss_gan_re_d",
        "is_wgan": False,
        "model_dir": "logs/ann_gan"
    }

    config_gru = {
        "params": {
            "layer_size": [32, dataset.get_input_shape()[-1]],
            "activation": 'tanh',
            "dropout": 0,
            "output_activation": "tanh",
            "cell_type": "gru",

        },
        "input_shape": dataset.get_input_shape(),
        "output_shape": dataset.get_output_shape(),
        "optimizer": 'adam',
        "learning_rate": 0.004,
        "model_dir": "test"
    }

    config_train = {
        "validation_split": 0.2,
        "batch_size": 32,
        "epochs": 10,
        "verbose": 1,
        "step_print": 1
    }
    function.run('GruModel', config_init=config_gru, config_train=config_train,
                 dataset=dataset, filename=None)


if __name__ == '__main__':
    test()
