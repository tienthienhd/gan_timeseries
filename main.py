from data import DataSets
from run import run

if __name__ == '__main__':
    dataset = DataSets('data/traffic/internet-traffic-data-in-bits-fr_EU_5m.csv',
                       usecols=[1],
                       column_names=['cpu'],
                       header=0,
                       n_in=4,
                       n_out=10,
                       is_diff=False,
                       is_log=False,
                       is_stand=False,
                       is_scale=True,
                       feature_range=(0, 1))

    config_ann = {
        "params": {
            "layer_size": [32, dataset.get_output_shape()[0] * dataset.get_output_shape()[1]],
            "activation": 'tanh',
            "dropout": 0.5,
            "output_activation": None

        },
        "input_shape": dataset.get_input_shape(),
        "output_shape": dataset.get_output_shape(),
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
        "is_wgan": False,
        "model_dir": "logs/flnn_gan"
    }

    config_ann_gan = {
        "params_generator": {
            "layer_size": [32, dataset.get_output_shape()[0] * dataset.get_output_shape()[1]],
            "activation": 'sigmoid',
            "dropout": 0,
            "concat_noise": "after",
            "output_activation": None

        },
        "params_discriminator": {
            "layer_size": [4, dataset.get_output_shape()[0] * dataset.get_output_shape()[1]],
            "activation": 'sigmoid',
            "dropout": 0,
            "output_activation": "sigmoid"

        },
        "input_shape": dataset.get_input_shape(),
        "output_shape": dataset.get_output_shape(),
        "noise_shape": dataset.get_input_shape(),
        "optimizer_g": 'rmsprop',
        "optimizer_d": 'rmsprop',
        "learning_rate_g": 0.001,
        "learning_rate_d": 0.001,
        "num_train_d": 1,
        "is_wgan": False,
        "model_dir": "logs/ann_gan"
    }

    config_gru_gan = {
        "params_generator": {
            "layer_size": [32, dataset.get_input_shape()[-1]],
            "activation": 'tanh',
            "dropout": 0,
            "output_activation": "tanh",
            "cell_type": "gru",
            "concat_noise": "after"

        },
        "params_discriminator": {
            "layer_size": [4, dataset.get_input_shape()[-1]],
            "activation": 'tanh',
            "dropout": 0,
            "output_activation": 'sigmoid',
            "cell_type": "gru"

        },
        "input_shape": dataset.get_input_shape(),
        "noise_shape": dataset.get_input_shape(),
        "optimizer_g": 'rmsprop',
        "optimizer_d": 'rmsprop',
        "learning_rate_g": 0.001,
        "learning_rate_d": 0.001,
        "num_train_d": 2,
        "is_wgan": False,
        "model_dir": "logs/ann_gan"
    }

    config_train = {
        "validation_split": 0.2,
        "batch_size": 8,
        "epochs": 10,
        "verbose": 1,
        "step_print": 1
    }
    run('AnnGan', config_init=config_ann_gan, config_train=config_train,
        dataset=dataset)
