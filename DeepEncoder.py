
import os
import math
import sys
import importlib

import numpy as np

import pandas as pd

from sklearn import linear_model
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, LabelBinarizer, RobustScaler, StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

from scipy.stats import norm

import tensorflow as tf
from tensorflow.python.client import device_lib

from plotnine import *

import matplotlib.pyplot as plt
from loss_history import LossHistory
import keras
from keras import backend as bkend
from keras_applications import vgg16
from keras.layers import UpSampling2D, BatchNormalization, Conv2D, MaxPooling2D
from keras.models import Model, Input, InputLayer
from keras.optimizers import Adam
from get_session import get_session
import keras.backend.tensorflow_backend as KTF
import inspect
KTF.set_session(get_session(gpu_fraction=0.75, allow_soft_placement=True, log_device_placement=False))


class AutoEncoder:
    def __init__(self):
        args, _, _, values = inspect.getargvalues(inspect.currentframe())
        values.pop("self")

        for arg, val in values.items():
            setattr(self, arg, val)

    @staticmethod
    def encoder(input_img):
        # encoder
        # input = 28 x 28 x 1 (wide and thin)
        conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)  # 28 x 28 x 32
        conv1 = BatchNormalization()(conv1)
        conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
        conv1 = BatchNormalization()(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)  # 14 x 14 x 32
        conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)  # 14 x 14 x 64
        conv2 = BatchNormalization()(conv2)
        conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
        conv2 = BatchNormalization()(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)  # 7 x 7 x 64
        conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)  # 7 x 7 x 128 (small and thick)
        conv3 = BatchNormalization()(conv3)
        conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
        conv3 = BatchNormalization()(conv3)
        conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv3)  # 7 x 7 x 256 (small and thick)
        conv4 = BatchNormalization()(conv4)
        conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
        conv4 = BatchNormalization()(conv4)
        return conv4

    @staticmethod
    def decoder(conv4):
        # decoder
        conv5 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv4)  # 7 x 7 x 128
        conv5 = BatchNormalization()(conv5)
        conv5 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv5)
        conv5 = BatchNormalization()(conv5)
        conv6 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv5)  # 7 x 7 x 64
        conv6 = BatchNormalization()(conv6)
        conv6 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv6)
        conv6 = BatchNormalization()(conv6)
        up1 = UpSampling2D((2, 2))(conv6)  # 14 x 14 x 64
        conv7 = Conv2D(32, (3, 3), activation='relu', padding='same')(up1)  # 14 x 14 x 32
        conv7 = BatchNormalization()(conv7)
        conv7 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv7)
        conv7 = BatchNormalization()(conv7)
        up2 = UpSampling2D((2, 2))(conv7)  # 28 x 28 x 32
        decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(up2)  # 28 x 28 x 1
        return decoded

    def build(self, input_img, loss_fn='mean_squared_error'):
        autoencoder = Model(input_img, self.decoder(self.encoder(input_img)))
        autoencoder.compile(loss=loss_fn, optimizer=Adam())
        return autoencoder

    @staticmethod
    def _get_calbacks():
        loss_history = LossHistory()
        early_stop = keras.callbacks.EarlyStopping(monitor="val_loss",
                                                   patience=10)
        reduce_learn_rate = keras.callbacks.ReduceLROnPlateau(monitor="val_loss",
                                                              factor=0.1,
                                                              patience=20)

        return [loss_history, early_stop, reduce_learn_rate]

    def train_autoencoder(self, autoencoder: Model):

        train_X, valid_X, train_ground, valid_ground = train_test_split(train_data,
                                                                        train_data,
                                                                        test_size=0.2,
                                                                        random_state=13)
        batch_size = 64
        epochs = 200
        inChannel = 1
        x, y = 28, 28
        input_img = Input(shape=(x, y, inChannel))
        num_classes = 10
        history = autoencoder.fit(train_X, train_ground, batch_size=batch_size, epochs=epochs,
                                  verbose=1, validation_data=(valid_X, valid_ground),
                                  callbacks=self._get_calbacks())
        return history

    @staticmethod
    def visualize(history):
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs = range(200)
        plt.figure()
        plt.plot(epochs, loss, 'b', label='Training loss')
        plt.plot(epochs, val_loss, 'r', label='Validation loss')
        plt.title('Training and validation loss')
        plt.legend()
        plt.show()
