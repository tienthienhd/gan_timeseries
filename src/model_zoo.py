from model import GanModel, RegressionModel
from net import *
import tensorflow as tf
import utils

__all__ = [
    "AnnModel",
    "AnnGan",
    "FlnnModel",
    "FlnnFlnnGan"

    # "LstmGan",
    # "GruGan",
    # "LstmCnnGan"
]


class AnnModel(RegressionModel):
    def __init__(self, params, input_shape, optimizer, learning_rate, model_dir='logs/ann'):
        net = DenseNet(params, 'dense_net')
        optimizer = utils.get_optimizer(optimizer, learning_rate)
        super().__init__(net, input_shape=input_shape, optimizer=optimizer, model_dir=model_dir)


class FlnnModel(RegressionModel):
    def __init__(self, params, input_shape, optimizer, learning_rate, model_dir='logs/flnn'):
        net = FlnnNet(params, 'flnn')
        optimizer = utils.get_optimizer(optimizer, learning_rate)
        super().__init__(net, input_shape, optimizer, model_dir)


class AnnGan(GanModel):
    def __init__(self,
                 params_generator,
                 params_discriminator,
                 input_shape,
                 noise_shape,
                 optimizer_g,
                 optimizer_d,
                 learning_rate_g,
                 learning_rate_d,
                 num_train_d,
                 is_wgan=False,
                 model_dir='logs/ann_gan'):
        generator = DenseNet(params_generator, 'generator')
        discriminator = DenseNet(params_discriminator, 'discriminator')

        optimizer_g = utils.get_optimizer(optimizer_g, learning_rate_g)
        optimizer_d = utils.get_optimizer(optimizer_d, learning_rate_d)

        super().__init__(generator, discriminator, input_shape, noise_shape, optimizer_g, optimizer_d, num_train_d,
                         model_dir, is_wgan)


class FlnnGan(GanModel):
    def __init__(self,
                 params_generator,
                 params_discriminator,
                 input_shape,
                 noise_shape,
                 optimizer_g,
                 optimizer_d,
                 learning_rate_g,
                 learning_rate_d,
                 num_train_d,
                 is_wgan=False,
                 model_dir='logs/ann_gan'):
        generator = FlnnNet(params_generator, 'generator')
        discriminator = FlnnNet(params_discriminator, 'discriminator')

        optimizer_g = utils.get_optimizer(optimizer_g, learning_rate_g)
        optimizer_d = utils.get_optimizer(optimizer_d, learning_rate_d)

        super().__init__(generator, discriminator, input_shape, noise_shape, optimizer_g, optimizer_d, num_train_d,
                         model_dir, is_wgan)


class GruGan(GanModel):
    def __init__(self,
                 params_generator,
                 params_discriminator,
                 input_shape,
                 noise_shape,
                 optimizer_g,
                 optimizer_d,
                 learning_rate_g,
                 learning_rate_d,
                 num_train_d,
                 is_wgan=False,
                 model_dir='logs/ann_gan'):
        generator = RnnNet(params_generator, 'generator')
        discriminator = RnnNet(params_discriminator, 'discriminator')

        optimizer_g = utils.get_optimizer(optimizer_g, learning_rate_g)
        optimizer_d = utils.get_optimizer(optimizer_d, learning_rate_d)

        super().__init__(generator, discriminator, input_shape, noise_shape, optimizer_g, optimizer_d, num_train_d,
                         model_dir, is_wgan)
