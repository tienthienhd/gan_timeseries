from data import DataSets
from run import run

if __name__ == '__main__':
    dataset = DataSets('data/gg_trace/5.csv',
                       usecols=[3],
                       column_names=['cpu'],
                       header=None,
                       n_in=1,
                       n_out=1,
                       is_diff=True,
                       feature_range=(-1, 1))

    config_ann = {
        "model": {
            "params": {
                "layer_size": [4, dataset.get_input_shape()[-1]],
                "activation": 'tanh',
                "dropout": 0,
                "output_activation": None

            },
            "input_shape": dataset.get_input_shape(),
            "optimizer": 'adam',
            "learning_rate": 0.001,
            "model_dir": "test"
        },
        "train": {
            "validation_split": 0.2,
            "batch_size": 8,
            "epochs": 100,
            "verbose": 1,
            "step_print": 1
        }
    }

    config_flnn = {
        "model": {
            "params": {
                "list_function": ['sin', 'cos', 'tan'],
                "activation": 'tanh',
                "num_output": 1

            },
            "input_shape": dataset.get_input_shape(),
            "optimizer": 'adam',
            "learning_rate": 0.001,
            "model_dir": "test"
        },
        "train": {
            "validation_split": 0.2,
            "batch_size": 8,
            "epochs": 10,
            "verbose": 1,
            "step_print": 1
        }
    }

    config_flnn_gan = {
        "model": {
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
            "noise_shape": dataset.get_input_shape(),
            "optimizer_g": 'adam',
            "optimizer_d": 'adam',
            "learning_rate_g": 0.001,
            "learning_rate_d": 0.001,
            "num_train_d": 1,
            "is_wgan": False,
            "model_dir": "logs/flnn_gan"
        },
        "train": {
            "validation_split": 0.2,
            "batch_size": 8,
            "epochs": 1000,
            "verbose": 1,
            "step_print": 10
        }
    }

    config_ann_gan = {
        "model": {
            "params_generator": {
                "layer_size": [32, dataset.get_input_shape()[-1]],
                "activation": 'sigmoid',
                "dropout": 0,
                "output_activation": None

            },
            "params_discriminator": {
                "layer_size": [32, dataset.get_input_shape()[-1]],
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
            "num_train_d": 1,
            "is_wgan": False,
            "model_dir": "logs/ann_gan"
        },
        "train": {
            "validation_split": 0.2,
            "batch_size": 8,
            "epochs": 3,
            "verbose": 1,
            "step_print": 1
        }
    }

    run('AnnGan', config=config_ann_gan, dataset=dataset)
