import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense, Conv2D, Flatten, BatchNormalization
from keras.optimizers import Adam
from keras.initializers import RandomUniform, VarianceScaling, Orthogonal
from keras.callbacks import TensorBoard
import tensorflow as tf

def huber_loss(y_true, y_pred):
    return tf.losses.huber_loss(y_true, y_pred)

class Dqn:
    """Implements a Deep Q Network"""

    # pylint: disable=too-many-instance-attributes

    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.model = Sequential()
        self.model.add(BatchNormalization(input_shape=self.state_size))
        self.model.add(Conv2D(32,
                         8,
                         strides=(4, 4),
                         padding="valid",
                         activation="relu",
                         input_shape=self.state_size,
                         kernel_initializer=Orthogonal,
                         use_bias=False,
                         data_format="channels_first"))
        self.model.add(Conv2D(64,
                         4,
                         strides=(2, 2),
                         padding="valid",
                         activation="relu",
                         input_shape=self.state_size,
                         kernel_initializer=VarianceScaling(scale=2),
                         use_bias=False,
                         data_format="channels_first"))
        self.model.add(Conv2D(64,
                         3,
                         strides=(1, 1),
                         padding="valid",
                         activation="relu",
                         input_shape=self.state_size,
                         kernel_initializer=VarianceScaling(scale=2),
                         use_bias=False,
                         data_format="channels_first"))
        # self.model.add(Conv2D(512,
        #                  7,
        #                  strides=(1, 1),
        #                  padding="valid",
        #                  activation="relu",
        #                  input_shape=self.state_size,
        #                  kernel_initializer=VarianceScaling(scale=2),
        #                  use_bias=False,
        #                  data_format="channels_first"))

        self.model.add(Flatten())
        self.model.add(Dense(512, activation="relu", use_bias=False, kernel_initializer=VarianceScaling(scale=2)))
        self.model.add(Dense(self.action_size, kernel_initializer=VarianceScaling(scale=2), use_bias=False))
        self.model.compile(optimizer=Adam(), loss=huber_loss, metrics=['accuracy'])