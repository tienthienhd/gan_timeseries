import math

import tensorflow as tf
import numpy as np
from metrics import evaluate


class Model:
    def __init__(self, model_dir):
        self.model_dir = model_dir

        tf.reset_default_graph()
        self.sess = tf.Session()

        self._build_model()
        # self.sess.run(tf.local_variables_initializer())
        self.sess.run(tf.global_variables_initializer())

    def close_session(self):
        self.sess.close()

    def _build_model(self):
        pass

    def _step(self, x, y=None, mode='predict'):
        pass

    def _step_batch(self, x, y=None, batch_size=1, mode='predict'):
        n_batch = math.ceil(len(x) / batch_size)
        results = []
        for batch in range(n_batch):
            x_b = x[batch * batch_size: (batch + 1) * batch_size]
            if y is not None:
                y_b = y[batch * batch_size: (batch + 1) * batch_size]
            else:
                y_b = None
            res = self._step(x_b, y_b, mode)
            results.append(res)
        if mode == 'predict':
            return np.concatenate(results, axis=0)

    def fit(self, x, y, validation_split=0, batch_size=1, epochs=1, verbose=1, step_print=1):
        do_validation = False
        if 0 < validation_split < 1:
            do_validation = True
            n_train = int((1 - validation_split) * len(x))
            x_val = x[n_train:]
            y_val = y[n_train:]
            x_train = x[:n_train]
            y_train = y[:n_train]
        else:
            x_train = x
            y_train = y
        if verbose > 0:
            print("Train on datasets: x={}; y={}".format(x.shape, y.shape))
            if do_validation:
                print("Datasets had been split to train [{}-{}] and val [{} - {}]".format(x_train.shape, y_train.shape,
                                                                                          x_val.shape, y_val.shape))
        for epoch in range(epochs):
            if verbose > 0 and (epoch + 1) % step_print == 0:
                print("Epoch {}/{}".format(epoch + 1, epochs), end=":")
            self._step_batch(x_train, y_train, batch_size, mode='train')
            if do_validation and verbose > 0:
                result_eval = self.evaluate(x_val, y_val)
                if verbose > 0 and (epoch + 1) % step_print == 0:
                    print(result_eval)

    def predict(self, x):
        return self._step_batch(x, mode='predict')

    def evaluate(self, x, y):
        pred = self.predict(x)
        return evaluate(y, pred, ('mae', 'rmse', 'mape', 'smape'))


class RegressionModel(Model):

    def __init__(self, net, input_shape, output_shape, optimizer, model_dir):
        self.net = net
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.optimizer = optimizer
        super().__init__(model_dir)

    def _build_model(self):
        self._x = tf.placeholder(tf.float32, [None] + self.input_shape, 'x')

        self._pred = self.net(self._x)
        self._pred = tf.reshape(self._pred, [-1] + self.output_shape)
        self._y = tf.placeholder(tf.float32, self._pred.shape, 'y')

        self._loss = tf.losses.mean_squared_error(self._y, self._pred)
        self._train_op = self.optimizer.minimize(self._loss)

    def _step(self, x, y=None, mode='predict'):
        if mode == 'predict':
            return self.sess.run(self._pred, feed_dict={self._x: x})
        elif mode == 'train':
            l, _ = self.sess.run([self._loss, self._train_op], feed_dict={self._x: x, self._y: y})
            return l


class GanModel(Model):

    def __init__(self, generator, discriminator, input_shape, output_shape, noise_shape, optimizer_g, optimizer_d, num_train_d,
                 model_dir, is_wgan=False):
        self.generator = generator
        self.discriminator = discriminator
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.noise_shape = noise_shape
        self.optimizer_g = optimizer_g
        self.optimizer_d = optimizer_d
        self.num_train_d = num_train_d
        self.is_wgan = is_wgan
        self.n_gen = 10
        self.c = 0.01
        super().__init__(model_dir)

    def _build_model(self):
        self._x = tf.placeholder(tf.float32, [None] + self.input_shape, 'x')
        self._z = tf.placeholder(tf.float32, [self.n_gen, None] + self.noise_shape, 'noise')

        # self._pred = self.generator(x=self._x, z=self._z[0])
        # self._pred = tf.reshape(self._pred, [-1] + self.output_shape)

        predicts = []
        for i in range(self.n_gen):
            p = self.generator(x=self._x, z=self._z[i])
            p = tf.reshape(p, [-1] + self.output_shape)
            predicts.append(p)

        self._pred = tf.math.reduce_mean(predicts, axis=0)
        # self._pred_std = tf.math.reduce_std(pred)

        self._y = tf.placeholder(tf.float32, self._pred.shape, 'y')

        # _pred = tf.reshape(self._pred, (-1, 1, self._pred.shape[-1]))
        # _y = tf.reshape(self._y, (-1, 1, self._y.shape[-1]))
        x_fake = tf.concat([self._x, self._pred], axis=1, name='x_fake')
        x_real = tf.concat([self._x, self._y], axis=1, name='x_real')

        d_fake = self.discriminator(x_fake, reuse=False)
        d_real = self.discriminator(x_real, reuse=True)

        if self.is_wgan:
            self._loss_g, self._loss_d = self._loss_wgan(d_fake, d_real)
        else:
            # self._loss_g, self._loss_d = self._loss_gan(d_fake, d_real)
            # self._loss_g = tf.Print(self._loss_g, [self._loss_g], message="loss_g before")
            # self._loss_g += tf.losses.mean_squared_error(self._y, self._pred)
            # self._loss_g = tf.Print(self._loss_g, [self._loss_g], message="loss_g after")
            self._loss_g, self._loss_d = self.custom_loss(d_fake, d_real, self._pred, self._y)

        d_vars, self._train_d = self._train_op(self._loss_d, self.optimizer_d, scope='discriminator')
        g_vars, self._train_g = self._train_op(self._loss_g, self.optimizer_g, scope='generator')

        if self.is_wgan:
            self.clip_d = [p.assign(tf.clip_by_value(p, -self.c, self.c)) for p in d_vars]

    def _loss_gan(self, d_fake, d_real):
        loss_d_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=d_fake, labels=tf.zeros_like(d_fake)))
        loss_d_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=d_real, labels=tf.ones_like(d_real)))
        loss_d = loss_d_real + loss_d_fake

        loss_g = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=d_fake,
                                                    labels=tf.ones_like(
                                                        d_fake)))
        return loss_g, loss_d

    def _loss_wgan(self, d_fake, d_real):
        loss_d = tf.reduce_mean(d_real) - tf.reduce_mean(d_fake)
        loss_g = -tf.reduce_mean(d_fake)
        return loss_g, loss_d

    def custom_loss(self, d_fake, d_real, mean_predict, actual):
        loss_g_gan, loss_d_gan = self._loss_gan(d_fake, d_real)
        # std_predict = tf.math.reduce_std(predicts, axis=0)

        loss_regression = tf.losses.mean_squared_error(actual, mean_predict)
        # loss_std = tf.reduce_mean(tf.square(std_predict))
        return loss_g_gan + loss_regression, loss_d_gan

    def _train_op(self, loss, optimizer, scope):
        var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)
        train_op = optimizer.minimize(loss, var_list=var_list)
        return var_list, train_op

    def _step(self, x, y=None, mode='predict'):
        if mode == 'predict':
            return self.sess.run(self._pred, feed_dict={self._x: x, self._z: self._get_noise(len(x))})
        elif mode == 'train':
            for i in range(self.num_train_d):
                ld, _ = self.sess.run([self._loss_d, self._train_d],
                                      {self._x: x, self._z: self._get_noise(len(x)), self._y: y})
                if self.is_wgan:
                    self.sess.run(self.clip_d)
            lg, _ = self.sess.run([self._loss_g, self._train_g],
                                  {self._x: x, self._z: self._get_noise(len(x)), self._y: y})

    def _get_noise(self, batch_size, loc=0, scale=1):
        noise_shape = [self.n_gen, batch_size] + self.noise_shape
        return np.random.normal(loc=loc, scale=scale, size=noise_shape)
