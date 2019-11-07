from data import DataSets
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from metrics import evaluate
import model_zoo
import gc

__all__ = [
    "fitness_function"
]

template_space = [
    {'name': 'n_in', 'type': 'discrete', 'domain': [1, 2, 3, 4, 5, 6, 7, 8]},
    {'name': 'g_layer_size', 'type': 'discrete', 'domain': [2, 4, 8, 16, 32, 64]},
    {'name': 'g_dropout', 'type': 'continuous', 'domain': (0, 0.8)},

    {'name': 'd_layer_size', 'type': 'discrete', 'domain': [2, 4, 8, 16, 32, 64]},
    {'name': 'd_dropout', 'type': 'continuous', 'domain': (0, 0.8)},
    {'name': 'learning_rate', 'type': 'continuous', 'domain': (0.0001, 0.1)},
    {'name': 'learning_rate', 'type': 'continuous', 'domain': (0.0001, 0.1)},
    {'name': 'num_train_d', 'type': 'discrete', 'domain': [1, 2, 3, 4, 5]},
    {'name': 'batch_size', 'type': 'discrete', 'domain': [4, 8, 16, 32]},

]

template_param = [1, 2, 0.3, 2, 0.1, 0.001, 0.001, 1, 1000]


def fitness_function(param):
    print("RUN PARAM:", param)
    model_name = "GruGan"

    n_in = int(param[0])
    n_out = 1

    data = DataSets("../../../data/gg_trace/5.csv",
                    usecols=[3],
                    column_names=['cpu'],
                    header=None,
                    n_in=n_in,
                    n_out=n_out,
                    is_diff=True,
                    is_log=True,
                    is_stand=True,
                    is_scale=True,
                    feature_range=(-1, 1))

    layer_size_g = int(param[1])
    activation_g = 'tanh'
    dropout_g = param[2]
    output_activation_g = 'tanh'
    cell_type_g = 'gru'
    concat_noise = 'after'

    layer_size_d = int(param[3])
    activation_d = 'tanh'
    dropout_d = param[4]
    output_activation_d = 'sigmoid'
    cell_type_d = 'gru'

    input_shape = data.get_input_shape()
    output_shape = data.get_output_shape()
    noise_shape = data.get_input_shape()
    optimizer_g = 'adam'
    optimizer_d = 'adam'
    learning_rate_g = param[5]
    learning_rate_d = param[6]
    num_train_d = int(param[7])
    is_wgan = False
    model_dir = 'logs/gan/'

    validation_split = 0.2
    batch_size = int(param[8])
    epochs = 10
    verbose = 0
    step_print = 1

    config_init = {
        "params_generator": {
            "layer_size": [layer_size_g, data.get_input_shape()[-1]],
            "activation": activation_g,
            "dropout": dropout_g,
            "output_activation": output_activation_g,
            "cell_type": cell_type_g,
            "concat_noise": concat_noise

        },
        "params_discriminator": {
            "layer_size": [layer_size_d, data.get_input_shape()[-1]],
            "activation": activation_d,
            "dropout": dropout_d,
            "output_activation": output_activation_d,
            "cell_type": cell_type_d

        },
        "input_shape": input_shape,
        "output_shape": output_shape,
        "noise_shape": noise_shape,
        "optimizer_g": optimizer_g,
        "optimizer_d": optimizer_d,
        "learning_rate_g": learning_rate_g,
        "learning_rate_d": learning_rate_d,
        "num_train_d": num_train_d,
        "is_wgan": is_wgan,
        "model_dir": model_dir
    }

    config_train = {
        "validation_split": validation_split,
        "batch_size": batch_size,
        "epochs": epochs,
        "verbose": verbose,
        "step_print": step_print
    }

    filename = "/home/tienthien/Desktop/Mine/gan_timeseries/logs/tuning/gru_gan/"
    for k, v in config_init.items():
        if k == 'model_dir':
            continue
        if not isinstance(v, dict):
            filename += "{}_".format(v)
        else:
            for k1, v1 in v.items():
                filename += "{}_".format(v1)

    res = run(model_name, config_init, config_train, data, filename)
    del data
    gc.collect()
    return res


def run(model, config_init, config_train, dataset: DataSets, filename, plot_pred=True, plot_dis=False):
    x_train, x_test, y_train, y_test = dataset.get_data()

    model = getattr(model_zoo, model)(**config_init)
    model.fit(x_train, y_train, **config_train)

    preds = []
    for i in range(10):
        pred = np.reshape(model.predict(x_test), (-1, y_test.shape[-1]))
        preds.append(pred)

    preds_invert = []
    for pred in preds:
        preds_invert.append(dataset.invert_transform(pred))

    model.close_session()

    preds_invert = np.concatenate(preds_invert, axis=1)
    pred_mean = np.mean(preds_invert, axis=1)
    pred_std = np.std(preds_invert, axis=1)

    y_test = np.reshape(y_test, (-1, y_test.shape[-1]))
    actual_invert = dataset.invert_transform(y_test)

    result_eval = evaluate(actual_invert, pred_mean, ["mae", 'smape', 'jsd'])

    df = pd.DataFrame(np.concatenate([actual_invert, preds_invert], axis=1),
                      columns=['actual'] + [f"predict{i}" for i in range(10)])
    df.to_csv(filename + ".csv", index=False)

    if plot_pred:
        plot_predict(actual_invert, pred_mean, pred_std, title=str(result_eval), path=filename)
    if plot_dis:
        plot_distribution(actual_invert, pred_mean, title=str(result_eval), path=None)
    del model, preds_invert, pred_mean, pred_std, df, dataset
    return result_eval['mae']


def plot_predict(actual, pred_mean, pred_std, title=None, path=None):
    plt.figure()

    pred_upper_95 = pred_mean + 1.96 * pred_std
    pred_lower_95 = pred_mean - 1.96 * pred_std

    pred_upper_80 = pred_mean + 1.28 * pred_std
    pred_lower_80 = pred_mean - 1.28 * pred_std

    plt.fill_between(range(len(pred_mean)), pred_lower_95, pred_upper_95, alpha=0.8, label='interval 95%')
    plt.fill_between(range(len(pred_mean)), pred_lower_80, pred_upper_80, alpha=0.5, label='interval 80%')

    plt.plot(actual, label='actual', color='#f48024')
    plt.plot(pred_mean, label='predict', color='green')
    plt.legend()
    if title is not None:
        plt.title(title)
    if path is not None:
        plt.savefig(path+".svg", dpi=300, format='svg')
    else:
        plt.show()
    plt.clf()
    plt.close()


def plot_distribution(actual, predict, title=None, path=None):
    plt.figure()
    sns.distplot(actual, bins=100, label='actual', color='#f48024')
    sns.distplot(predict, bins=100, label='predict', color='green')
    plt.legend()
    if title is not None:
        plt.title(title)
    if path is not None:
        plt.savefig(path, dpi=300, format='svg')
    else:
        plt.show()
    plt.clf()
    plt.close()


def custom_fitness(params):
    res = []
    for param in params:
        res.append(fitness_function(param))
    return res


if __name__ == '__main__':
    print(fitness_function(template_param))
