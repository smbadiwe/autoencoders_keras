
import os
from loss_history import LossHistory
import keras
from keras.applications.vgg16 import VGG16  # 138,357,544
from keras.layers import UpSampling2D, Conv2D, MaxPooling2D, Deconv2D
from keras.models import Model, Input
from get_session import get_session
import keras.backend.tensorflow_backend as KTF
import inspect
import matplotlib.pyplot as plt
from data_generator import DataGenerator
from math import floor
import pylab
pylab.rcParams['figure.figsize'] = (8.0, 10.0)

KTF.set_session(get_session(gpu_fraction=0.75, allow_soft_placement=True, log_device_placement=False))


class AutoEncoder:
    """
    Auto-encode with VGG encoder and decoder, and COCO test2014 dataset.
    http://images.cocodataset.org/zips/test2014.zip (at http://cocodataset.org/#download)
    """
    def __init__(self):
        args, _, _, values = inspect.getargvalues(inspect.currentframe())
        values.pop("self")
        self.folder = "../../.keras/datasets/coco-test2014/test2014/"

        for arg, val in values.items():
            setattr(self, arg, val)

    @staticmethod
    def decoder(x):
        # with tf.device("cpu:0"):
        # # --------latent space (trainable) ------------
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='latent')(x)
        encoder_fn = VGG16(include_top=False, weights=None)
        for i in range(len(encoder_fn.layers) - 1, 0, -1):  # ignore the 1st (input) layer
            layer = encoder_fn.layers[i]
            layer.trainable = True
            if isinstance(layer, MaxPooling2D):
                x = UpSampling2D((2, 2))(x)
            else:  # in vgg, everything is Conv2D
                config = layer.get_config()
                config['name'] = "d_" + layer.name
                x = Deconv2D.from_config(config)(x)

        # finally, bring it back to input shape
        x = Deconv2D(3, (3, 3), activation='relu', padding='same', name='dblock1_conv3')(x)
        return x

    @staticmethod
    def encoder(input_img):
        # with tf.device("cpu:0"):
        encoder_fn = VGG16(include_top=False, weights='imagenet', input_tensor=input_img)
        x = None
        for layer in encoder_fn.layers[1:]:  # ignore the 1st (input) layer
            layer.trainable = True  # using pre-trained weights for the encoder
            if x is None:
                x = layer(input_img)
            else:
                x = layer(x)
        return x

    def build(self, input_img, loss_fn='mean_squared_error'):
        m = Model(input_img, self.decoder(self.encoder(input_img)))
        m.compile(loss=loss_fn, optimizer=keras.optimizers.RMSprop(clipnorm=1.))  # clipvalue=0.5))  #
        return m

    @staticmethod
    def _get_calbacks():
        loss_history = LossHistory()
        early_stop = keras.callbacks.EarlyStopping(monitor="val_loss",
                                                   patience=10)
        reduce_learn_rate = keras.callbacks.ReduceLROnPlateau(monitor="val_loss",
                                                              factor=0.1,
                                                              patience=20)
        nan_loss = keras.callbacks.TerminateOnNaN()
        return [loss_history, early_stop, reduce_learn_rate, nan_loss]

    def train_autoencoder(self, autoencoder: Model):
        train, val, test = self.load_dataset()
        print(f"training with {len(train)} samples")
        train_gen = DataGenerator(train, data_folder=self.folder)
        val_gen = DataGenerator(val, data_folder=self.folder)
        # test_gen = DataGenerator(test, data_folder=self.folder)
        history = autoencoder.fit_generator(train_gen, epochs=200,  # use_multiprocessing=True, workers=6,
                                            validation_data=val_gen,
                                            callbacks=self._get_calbacks())
        return history, test

    def load_dataset(self):
        for _, _, files in os.walk(self.folder):
            sorted(files)
            ln = len(files) * 0.1
            tr, va = floor(0.8 * ln), floor(0.95 * ln)
            return files[:tr], files[tr:va], files[va:]

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


if __name__ == "__main__":
    au = AutoEncoder()
    model = au.build(Input(shape=(224, 224, 3)))
    print(model.summary())
    hist, test_data = au.train_autoencoder(model)
    print("summarize history for loss")
    # summarize history for loss
    au.visualize(hist)
    # plt.plot(hist.history['loss'])
    # plt.plot(hist.history['val_loss'])
    # plt.title('model loss')
    # plt.ylabel('loss')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'test'], loc='upper left')
    # plt.show()
    # print(model.summary())
