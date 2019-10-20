from data import DataSets
import numpy as np
import pandas as pd
import utils
import model_zoo
from metrics import evaluate


def run(model, config, dataset: DataSets):
    x_train, x_test, y_train, y_test = dataset.get_data(0.2)

    model = getattr(model_zoo, model)(**config['model'])
    model.fit(x_train, y_train, **config['train'])

    pred = model.predict(x_test)
    model.close_session()

    pred = np.reshape(pred, (-1, y_test.shape[-1]))
    y_test = np.reshape(y_test, (-1, y_test.shape[-1]))

    pred_invert = dataset.invert_transform(pred)
    y_test_invert = dataset.invert_transform(y_test)

    result_eval = evaluate(y_test_invert, pred_invert, metrics=['mae', 'rmse', 'mape', 'smape'])

    print("Result test:", result_eval)

    short = ['actual', 'predict']
    long = ['actual_cpu', 'actual_mem', 'predict_cpu', 'predict_mem']
    columns = short

    a = np.concatenate([y_test, pred], axis=1)
    df = pd.DataFrame(a, columns=columns)
    utils.plot_distribution([df[col].values for col in columns], legends=columns)
    utils.plot_time_series(df)

    b = np.concatenate([y_test_invert, pred_invert], axis=1)
    df_2 = pd.DataFrame(b, columns=columns)
    utils.plot_distribution([df_2[col].values for col in columns], legends=columns)
    utils.plot_time_series(df_2, title=f"{result_eval}")
    utils.plt.show()


if __name__ == '__main__':
    dataset = DataSets('../data/gg_trace/5.csv',
                       usecols=[3],
                       column_names=['cpu'],
                       header=None,
                       n_in=1,
                       n_out=1)

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
            "batch_size": 4,
            "epochs": 100,
            "verbose": 1,
            "step_print": 1
        }
    }

    config_flnn = {
        "model": {
            "params": {
                "list_function": ['sin', 'cos', 'tan'],
                "activation": 'sigmoid',
                "num_output": 1

            },
            "input_shape": dataset.get_input_shape(),
            "optimizer": 'adam',
            "learning_rate": 0.001,
            "model_dir": "test"
        },
        "train": {
            "validation_split": 0.2,
            "batch_size": 4,
            "epochs": 100,
            "verbose": 1,
            "step_print": 1
        }
    }

    run('FlnnModel', config=config_flnn, dataset=dataset)
