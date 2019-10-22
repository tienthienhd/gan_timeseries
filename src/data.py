import math
import numpy as np
import utils
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split


class DataSets(object):
    def __init__(self,
                 data_path: str,
                 usecols: list = None,
                 column_names: list = None,
                 header=None,
                 n_in=1,
                 n_out=1,
                 test_size=0.2,
                 is_log=True,
                 is_diff=True,
                 is_stand=True,
                 is_scale=True,
                 feature_range=(0, 1),
                 debug=False,
                 show_plot=False, **kwargs):
        self.test_size=test_size
        self.is_log = is_log
        self.is_diff = is_diff
        self.is_stand = is_stand
        self.is_scale = is_scale
        self._data_raw = pd.read_csv(data_path, usecols=usecols, names=column_names, header=header)

        self.min_max_scaler = MinMaxScaler(feature_range=feature_range)
        self.standard_scaler = StandardScaler()

        history, data = self.transform(self._data_raw)
        self.history = history[n_in:]

        data_supervised = utils.make_data_supervise(data, n_in, n_out)
        data_supervised = np.reshape(data_supervised.values, (-1, n_in + n_out, self._data_raw.shape[-1]))
        self.x = data_supervised[:, :n_in]
        self.y = data_supervised[:, n_in:]

        self.history_train = None
        self.history_test = None

    def transform(self, data):
        x = data
        history = data
        if self.is_log:
            # reduce range of data
            data_ranged = self._reduce_range(x)
            x = data_ranged
            history = data_ranged
        if self.is_diff:
            # difference of data
            data_diff = self._difference(x, interval=1)
            x = data_diff

        if self.is_stand:
            data_standard = self.standard_scaler.fit_transform(x)
            x = data_standard
        if self.is_scale:
            data_scaled = self.min_max_scaler.fit_transform(x)
            x = data_scaled
        return history, x

    def invert_transform(self, data, history='test'):
        x = data
        if self.is_scale:
            x = self.min_max_scaler.inverse_transform(x)
        if self.is_stand:
            x = self.standard_scaler.inverse_transform(x)
        if self.is_diff:
            x = self._invert_difference(history, x, interval=1)
        if self.is_log:
            x = self._invert_range(x)
        return x

    def get_input_shape(self):
        return list(self.x.shape[1:])

    def get_data(self):
        test_size = self.test_size
        x = self.x
        y = self.y
        if test_size == 0:
            return x, y
        else:
            x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=False, test_size=test_size)
            self.history_train = self.history[:-len(x_test) - 1]
            self.history_test = self.history[-len(x_test) - 1:-1]
            return x_train, x_test, y_train, y_test

    def _reduce_range(self, data):
        """
        Apply log function to reduce range of data
        :param data:
        :return:
        """
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data)
        data_reduced = data.applymap(np.log1p)

        return data_reduced.values

    def _invert_range(self, data):
        """
        Invert data to origin range
        :param data:
        :return:
        """
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data)
        data = data.applymap(lambda x: np.exp(x) - 1)
        return data.values

    def _difference(self, data, interval=1):
        cols = None
        if isinstance(data, pd.DataFrame):
            cols = data.columns
            data = data.values
        diff = data[interval:] - data[:-interval]
        # diff = pd.DataFrame(diff, columns=cols)
        return diff

    def _invert_difference(self, history_type, data, interval=1):
        history = None
        if history_type == 'train':
            history = self.history_train
        elif history_type == 'test':
            history = self.history_test
        elif history_type == 'all':
            history = self.history

        if history.shape[1] != data.shape[1]:
            raise Exception("DataSets: history's shape {} not equal data's shape {}".format(history.shape, data.shape))
        cols = None
        if isinstance(data, pd.DataFrame):
            cols = data.columns
            data = data.values
        if isinstance(history, pd.DataFrame):
            cols = history.columns
            history = history.values
        if history.shape[0] == data.shape[0]:
            result = history + data
        else:
            result = history[:-interval] + data
        return result
