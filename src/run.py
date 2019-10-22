from data import DataSets
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import utils
import model_zoo
from metrics import evaluate


def run(model, config_init, config_train, dataset: DataSets):
    x_train, x_test, y_train, y_test = dataset.get_data()
    y_train = y_train.reshape((-1, y_train.shape[-1]))

    model = getattr(model_zoo, model)(**config_init)
    model.fit(x_train, y_train, **config_train)

    preds = []
    for i in range(50):
        pred = model.predict(x_test)
        # print(pred.shape)
        pred = np.reshape(pred, (-1, y_test.shape[-1]))
        pred = dataset.invert_transform(pred)
        preds.append(pred.values)
    model.close_session()

    y_test = np.reshape(y_test, (-1, y_test.shape[-1]))
    y_test = dataset.invert_transform(y_test)

    pred_concat = np.concatenate(preds, axis=1)
    pred = pred_concat.mean(axis=1)
    pred = np.expand_dims(pred, axis=1)
    print(pred.shape)

    result_eval = evaluate(y_test, pred, metrics=['mae', 'rmse', 'mape', 'smape'])

    print("Result test:", result_eval)

    plot(y_test, preds, pred)
    plot_distribution(y_test, preds, pred)
    plt.show()


def plot(actual, predicts, predict_mean):
    plt.figure()
    pred_concat = np.concatenate(predicts, axis=1)
    pred_min = np.amin(pred_concat, axis=1)
    pred_max = np.amax(pred_concat, axis=1)
    plt.fill_between(range(len(pred_concat)), pred_min, pred_max, alpha=0.6)
    # for p in predicts:
    #     plt.plot(p, color='blue', alpha=0.3)
    plt.plot(actual, label='actual', color='#f48024')
    plt.plot(predict_mean, label='predict', color='green')
    plt.legend()


def plot_distribution(actual, predicts, predict_mean):
    plt.figure()
    # for p in predicts:
    #     sns.distplot(p, bins=100, color='blue')
    sns.distplot(actual, bins=100, label='actual', color='#f48024')
    sns.distplot(predict_mean, bins=100, label='predict', color='green')
    plt.legend()
