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
    # y_train = y_train.reshape((-1, y_train.shape[-1]))

    model = getattr(model_zoo, model)(**config_init)
    model.fit(x_train, y_train, **config_train)

    pred = model.predict(x_test)
    # pred = np.reshape(pred, (-1, y_test.shape[-1]))
    #
    # y_test = np.reshape(y_test, (-1, y_test.shape[-1]))

    plt.figure()
    start = 0
    end = 2000
    step = 10
    if step == 1:
        plt.plot(pred[start:end, 0])
    for i in range(start, end, step):
        plt.plot(range(i, i+step), pred[i])
    plt.plot(range(start, end), y_test[start:end, 0, 0], label='actual')
    plt.legend()
    plt.show()
    return 1

    pred_invert = dataset.invert_transform(pred)
    y_test_invert = dataset.invert_transform(y_test)

    output = np.concatenate([pred, y_test, pred_invert, y_test_invert], axis=1)
    df = pd.DataFrame(output, columns=['predict', 'actual', 'predict_invert', 'actual_invert'])

    filename = "/home/tienthien/Desktop/Mine/gan_timeseries/logs/tuning/gru_gan/"
    for k, v in config_init.items():
        if k == 'model_dir':
            continue
        if not isinstance(v, dict):
            filename += "{}_".format(v)
        else:
            for k1, v1 in v.items():
                filename += "{}_".format(v1)
    filename += ".csv"
    df.to_csv(filename, index=False)

    result_metrics = evaluate(y_test_invert, pred_invert,
                              metrics=('mae', 'rmse'))

    return result_metrics['mae']


def run_test(model, config_init, config_train, dataset: DataSets):
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
        preds.append(pred)
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

    pred_mean = np.mean(pred_concat, axis=1)
    pred_std = np.std(pred_concat, axis=1)
    pred_mean = pred_mean.reshape(len(pred_mean))

    pred_upper_95 = pred_mean + 1.96 * pred_std
    pred_lower_95 = pred_mean - 1.96 * pred_std

    pred_upper_80 = pred_mean + 1.28 * pred_std
    pred_lower_80 = pred_mean - 1.28 * pred_std

    plt.fill_between(range(len(pred_concat)), pred_min, pred_max, alpha=0.6, label='min-max-range')
    plt.fill_between(range(len(pred_concat)), pred_lower_95, pred_upper_95, alpha=0.4, label='interval 95%')
    plt.fill_between(range(len(pred_concat)), pred_lower_80, pred_upper_80, alpha=0.4, label='interval 80%')
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
