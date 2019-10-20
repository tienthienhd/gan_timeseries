import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import seaborn as sns


def plot_time_series(data, title=None):
    if isinstance(data, np.ndarray):
        data = pd.DataFrame(data)

    data.plot()
    plt.legend()
    plt.title(title)
    # plt.show()


def plot_distribution_(data, title=None):
    if isinstance(data, np.ndarray):
        data = pd.DataFrame(data)

    data.plot.hist(bins=200, alpha=0.5)
    plt.legend()
    plt.title(title)


def plot_distribution(data, legends=None, title=None, file_save=None):
    colors = ['red', 'green', 'blue', 'black']
    fig = plt.figure()
    for i, (d, l) in enumerate(zip(data, legends)):
        sns.distplot(d, bins=100, label=l, color=colors[i])
    plt.legend()
    if title is not None:
        plt.title(title)
    if file_save is not None:
        plt.savefig(file_save)
        plt.close(fig)



def get_optimizer(name, learning_rate):
    if name == 'adam':
        return tf.train.AdamOptimizer(learning_rate)
    elif name == 'rmsprop':
        return tf.train.RMSPropOptimizer(learning_rate)
    else:
        raise NotImplementedError


def make_data_supervise(data, n_in, n_out, dropnan=True):
    """
    Frame a time series as a supervised learning dataset.
    Arguments:
        data: Sequence of observations as a list or NumPy array.
        n_in: Number of lag observations as input (X).
        n_out: Number of observations as output (y).
        dropnan: Boolean whether or not to drop rows with NaN values.
    Returns:
        Pandas DataFrame of series framed for supervised learning.
    """
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg
