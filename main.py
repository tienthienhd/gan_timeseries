from data import DataSets
import run
if __name__ == '__main__':
    dataset = DataSets('data/traffic/internet-traffic-data-in-bits-fr_EU_5m.csv',
                       usecols=[1],
                       column_names=['cpu'],
                       header=0,
                       n_in=1,
                       n_out=1,
                       is_diff=True,
                       is_log=True,
                       is_stand=True,
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
        "batch_size": 32,
        "epochs": 20,
        "verbose": 1,
        "step_print": 1
    }
    run('GruGan', config_init=config_gru_gan, config_train=config_train,
        dataset=dataset)
