# License
# Copyright 2018 Hamaad Musharaf Shah
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

import math
import inspect

from sklearn.base import BaseEstimator, TransformerMixin

import keras
from keras.layers import Input, Dense, BatchNormalization, Dropout, Lambda, add
from keras.models import Model, Sequential

import tensorflow

from loss_history import LossHistory


class VariationalAutoencoder(BaseEstimator,
                             TransformerMixin):
    def __init__(self,
                 n_feat=None,
                 n_epoch=None,
                 batch_size=None,
                 encoder_layers=None,
                 decoder_layers=None,
                 n_hidden_units=None,
                 encoding_dim=None,
                 denoising=None):
        args, _, _, values = inspect.getargvalues(inspect.currentframe())
        values.pop("self")

        for arg, val in values.items():
            setattr(self, arg, val)

        loss_history = LossHistory()

        early_stop = keras.callbacks.EarlyStopping(monitor="val_loss",
                                                   patience=10)

        reduce_learn_rate = keras.callbacks.ReduceLROnPlateau(monitor="val_loss",
                                                              factor=0.1,
                                                              patience=20)

        self.callbacks_list = [loss_history, early_stop, reduce_learn_rate]

        for i in range(self.encoder_layers):
            if i == 0:
                self.input_data = Input(shape=(self.n_feat,))
                self.encoded = BatchNormalization()(self.input_data)
                self.encoded = Dense(units=self.n_hidden_units, activation="elu")(self.encoded)
                self.encoded = Dropout(rate=0.5)(self.encoded)
            elif i > 0 and i < self.encoder_layers - 1:
                self.encoded = BatchNormalization()(self.encoded)
                self.encoded = Dense(units=self.n_hidden_units, activation="elu")(self.encoded)
                self.encoded = Dropout(rate=0.5)(self.encoded)
            elif i == self.encoder_layers - 1:
                self.encoded = BatchNormalization()(self.encoded)
                self.encoded = Dense(units=self.n_hidden_units, activation="elu")(self.encoded)

        self.mu = Dense(units=self.encoding_dim, activation="linear")(self.encoded)
        self.log_sigma = Dense(units=self.encoding_dim, activation="linear")(self.encoded)
        z = Lambda(self.sample_z, output_shape=(self.encoding_dim,))([self.mu, self.log_sigma])

        self.decoded_layers_dict = {}

        decoder_counter = 0

        for i in range(self.decoder_layers):
            if i == 0:
                self.decoded_layers_dict[decoder_counter] = BatchNormalization()
                decoder_counter += 1
                self.decoded_layers_dict[decoder_counter] = Dense(units=self.n_hidden_units, activation="elu")
                decoder_counter += 1
                self.decoded_layers_dict[decoder_counter] = Dropout(rate=0.5)

                self.decoded = self.decoded_layers_dict[decoder_counter - 2](z)
                self.decoded = self.decoded_layers_dict[decoder_counter - 1](self.decoded)
                self.decoded = self.decoded_layers_dict[decoder_counter](self.decoded)

                decoder_counter += 1
            elif i > 0 and i < self.decoder_layers - 1:
                self.decoded_layers_dict[decoder_counter] = BatchNormalization()
                decoder_counter += 1
                self.decoded_layers_dict[decoder_counter] = Dense(units=self.n_hidden_units, activation="elu")
                decoder_counter += 1
                self.decoded_layers_dict[decoder_counter] = Dropout(rate=0.5)

                self.decoded = self.decoded_layers_dict[decoder_counter - 2](self.decoded)
                self.decoded = self.decoded_layers_dict[decoder_counter - 1](self.decoded)
                self.decoded = self.decoded_layers_dict[decoder_counter](self.decoded)

                decoder_counter += 1
            elif i == self.decoder_layers - 1:
                self.decoded_layers_dict[decoder_counter] = BatchNormalization()
                decoder_counter += 1
                self.decoded_layers_dict[decoder_counter] = Dense(units=self.n_hidden_units, activation="elu")

                self.decoded = self.decoded_layers_dict[decoder_counter - 1](self.decoded)
                self.decoded = self.decoded_layers_dict[decoder_counter](self.decoded)
                decoder_counter += 1

        # Output would have shape: (batch_size, n_feat).
        self.decoded_layers_dict[decoder_counter] = Dense(units=self.n_feat, activation="sigmoid")
        self.decoded = self.decoded_layers_dict[decoder_counter](self.decoded)

        self.autoencoder = Model(self.input_data, self.decoded)
        self.autoencoder.compile(optimizer=keras.optimizers.Adam(),
                                 loss=self.vae_loss)

    def fit(self,
            X,
            y=None):
        self.autoencoder.fit(X if self.denoising is None else X + self.denoising, X,
                             validation_split=0.3,
                             epochs=self.n_epoch,
                             batch_size=self.batch_size,
                             shuffle=True,
                             callbacks=self.callbacks_list,
                             verbose=1)

        self.encoder = Model(self.input_data, self.mu)

        self.generator_input = Input(shape=(self.encoding_dim,))
        self.generator_output = None
        decoder_counter = 0

        for i in range(self.decoder_layers):
            if i == 0:
                self.generator_output = self.decoded_layers_dict[decoder_counter](self.generator_input)
                decoder_counter += 1
                self.generator_output = self.decoded_layers_dict[decoder_counter](self.generator_output)
                decoder_counter += 1
                self.generator_output = self.decoded_layers_dict[decoder_counter](self.generator_output)
                decoder_counter += 1
            elif i > 0 and i < self.decoder_layers - 1:
                self.generator_output = self.decoded_layers_dict[decoder_counter](self.generator_output)
                decoder_counter += 1
                self.generator_output = self.decoded_layers_dict[decoder_counter](self.generator_output)
                decoder_counter += 1
                self.generator_output = self.decoded_layers_dict[decoder_counter](self.generator_output)
                decoder_counter += 1
            elif i == self.decoder_layers - 1:
                self.generator_output = self.decoded_layers_dict[decoder_counter](self.generator_output)
                decoder_counter += 1
                self.generator_output = self.decoded_layers_dict[decoder_counter](self.generator_output)
                decoder_counter += 1

        self.generator_output = self.decoded_layers_dict[decoder_counter](self.generator_output)

        self.generator = Model(self.generator_input, self.generator_output)

        return self

    def transform(self,
                  X):
        return self.encoder.predict(X)

    def sample_z(self,
                 args):
        mu_, log_sigma_ = args
        eps = keras.backend.random_normal(shape=(keras.backend.shape(mu_)[0], self.encoding_dim),
                                          mean=0.0,
                                          stddev=1.0)
        out = mu_ + keras.backend.exp(log_sigma_ / 2) * eps

        return out

    def vae_loss(self,
                 y_true,
                 y_pred):
        recon = keras.backend.sum(x=keras.backend.square(y_pred - y_true))
        kl = -0.5 * keras.backend.sum(
            x=1.0 + self.log_sigma - keras.backend.exp(self.log_sigma) - keras.backend.square(self.mu))
        return recon + kl
