import tensorflow as tf


class Net:
    def __init__(self, params, scope):
        self.params = params
        self.scope = scope

    def __call__(self, x, reuse=False, *args, **kwargs):
        pass


class DenseNet(Net):

    def __init__(self, params, scope):
        super().__init__(params, scope)
        self.layer_size = params['layer_size']
        self.activation = params['activation']
        self.dropout = params['dropout']
        self.output_activation = params['output_activation']

    def __call__(self, x, reuse=False, *args, **kwargs):
        with tf.variable_scope(self.scope) as scope:
            if reuse:
                scope.reuse_variables()
            net = tf.keras.layers.Flatten()(x)
            for i, units in enumerate(self.layer_size[:-1]):
                net = tf.keras.layers.Dense(units, self.activation)(net)
                net = tf.keras.layers.Dropout(self.dropout)(net)
            net = tf.keras.layers.Dense(self.layer_size[-1], activation=self.output_activation)(net)
        return net


class LstmNet(Net):
    def __init__(self, params, scope):
        super().__init__(params, scope)
        self.layer_size = params['layer_size']
        self.activation = params['activation']
        self.dropout = params['dropout']
        self.output_activation = params['output_activation']

    def __call__(self, x, reuse=False, *args, **kwargs):
        with tf.variable_scope(self.scope) as scope:
            if reuse:
                scope.reuse_variables()
            cells = []
            for i, units in enumerate(self.layer_size[:-1]):
                cell = tf.keras.layers.LSTMCell(units, activation=self.activation, dropout=self.dropout)
                cells.append(cell)
            net = tf.keras.layers.RNN(cells, return_sequences=False)(x)
            net = tf.keras.layers.Dense(self.layer_size[-1], activation=self.output_activation)(net)
        return net


class GruNet(Net):

    def __init__(self, params, scope):
        super().__init__(params, scope)
        self.layer_size = params['layer_size']
        self.activation = params['activation']
        self.dropout = params['dropout']
        self.output_activation = params['output_activation']

    def __call__(self, x, reuse=False, *args, **kwargs):
        with tf.variable_scope(self.scope) as scope:
            if reuse:
                scope.reuse_variables()
            cells = []
            for i, units in enumerate(self.layer_size[:-1]):
                cell = tf.keras.layers.GRUCell(units, activation=self.activation, dropout=self.dropout)
                cells.append(cell)
            net = tf.keras.layers.RNN(cells, return_sequences=False)(x)
            net = tf.keras.layers.Dense(self.layer_size[-1], activation=self.output_activation)(net)
        return net


class FlnnNet(Net):

    def __init__(self, params, scope):
        super().__init__(params, scope)
        self.activation = params['activation']
        self._get_function(params['list_function'])
        self.n_output = params['num_output']

    def _get_function(self, list_functions):
        functions = {
            'sin': tf.sin,
            'cos': tf.cos,
            'tan': tf.tan
        }
        self.functions = []
        for f_name in list_functions:
            f = functions[f_name]
            self.functions.append(f)

    def __call__(self, x, reuse=False, *args, **kwargs):
        net = [x]
        for f in self.functions:
            net.append(f(x))
        net = tf.concat(net, axis=1)
        net = tf.keras.layers.Flatten()(net)
        net = tf.keras.layers.Dense(self.n_output, activation=self.activation)(net)
        return net
