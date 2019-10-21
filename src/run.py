from data import DataSets
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import utils
import model_zoo
from metrics import evaluate


def run(model, config, dataset: DataSets):
    x_train, x_test, y_train, y_test = dataset.get_data(0.01)
    y_train = y_train.reshape((-1, y_train.shape[-1]))

    model = getattr(model_zoo, model)(**config['model'])
    model.fit(x_train, y_train, **config['train'])

    preds = []
    for i in range(100):
        pred = model.predict(x_test)
        # print(pred.shape)
        pred = np.reshape(pred, (-1, y_test.shape[-1]))
        pred_invert = dataset.invert_transform(pred)
        # print(pred_invert.shape)
        preds.append(pred_invert.values)
    model.close_session()

    y_test = np.reshape(y_test, (-1, y_test.shape[-1]))
    y_test_invert = dataset.invert_transform(y_test)

    pred_concat = np.concatenate(preds, axis=1)
    pred_invert = pred_concat.mean(axis=1)
    pred_invert = np.expand_dims(pred_invert, axis=1)
    print(pred_invert.shape)

    result_eval = evaluate(y_test_invert, pred_invert, metrics=['mae', 'rmse', 'mape', 'smape'])

    print("Result test:", result_eval)

    plot(y_test_invert, preds, pred_invert)
    plot_distribution(y_test_invert, preds, pred_invert)
    plt.show()

    # short = ['actual', 'predict']
    # long = ['actual_cpu', 'actual_mem', 'predict_cpu', 'predict_mem']
    # columns = short
    #
    # a = np.concatenate([y_test, pred], axis=1)
    # df = pd.DataFrame(a, columns=columns)
    # utils.plot_distribution([df[col].values for col in columns], legends=columns)
    # utils.plot_time_series(df)
    #
    # b = np.concatenate([y_test_invert, pred_invert], axis=1)
    # df_2 = pd.DataFrame(b, columns=columns)
    # utils.plot_distribution([df_2[col].values for col in columns], legends=columns)
    # utils.plot_time_series(df_2, title=f"{result_eval.values()}")
    # utils.plt.show()


def plot(actual, predicts, predict_mean):
    plt.figure()
    for p in predicts:
        plt.plot(p, color='blue', alpha=0.3)
    plt.plot(actual, label='actual', color='#f48024')
    plt.plot(predict_mean, label='predict', color='green')
    plt.legend()


def plot_distribution(actual, predicts, predict_mean):
    plt.figure()
    for p in predicts:
        sns.distplot(p, bins=100, color='blue')
    sns.distplot(actual, label='actual', color='#f48024')
    sns.distplot(predict_mean, label='predict', color='green')
    plt.legend()
