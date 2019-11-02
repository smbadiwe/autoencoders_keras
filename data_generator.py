import numpy as np
from keras.utils import Sequence
from os import path
from PIL import Image
from math import floor, ceil
import tensorflow as tf


class DataGenerator(Sequence):
    def __init__(self, filenames, data_folder, batch_size=32, img_shape=(224, 224, 3),
                 shuffle=False):
        self.img_shape = img_shape
        self.batch_size = batch_size
        self.data_folder = data_folder
        self.filenames = filenames
        self.shuffle = shuffle
        self.indexes = np.arange(len(filenames))
        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return floor(len(self.filenames) / self.batch_size)

    def __getitem__(self, index):
        """Generate one batch of data"""
        # with tf.device("cpu:0"):
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        batch = [self.filenames[k] for k in indexes]

        # Generate data
        X = self.__data_generation(batch)

        return X, X

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        # self.indexes = np.arange(len(self.filenames))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __crop(self, img):
        w, h, _ = img.shape
        out_w, out_h, _ = self.img_shape
        if out_h > w or out_w > w:
            # up-scaling not supported. This is just cropping.
            return img
        if w < h:
            center_h = floor(h/2)
            f = center_h - ceil(w/2)
            t = center_h + floor(w/2)
            img = img[:, f:t, :]
        elif w > h:
            center_w = floor(w/2)
            f = center_w - floor(h/2)
            t = center_w + ceil(h/2)
            img = img[f:t, :, :]
        if w != h:
            w, h, _ = img.shape  # this will already be square
        center = floor(w/2)
        if out_h == out_w:
            out_w = out_w/2
            x1, x2 = center - floor(out_w), center + ceil(out_w)
            return img[x1:x2, x1:x2, :]

        out_w, out_h = out_w/2, out_h/2
        x1, x2 = center - floor(out_w), center + ceil(out_w)
        y1, y2 = center - floor(out_h), center + ceil(out_h)
        return img[x1:x2, y1:y2, :]

    def __data_generation(self, filenames):
        """Generates data containing batch_size samples. X : (n_samples, *dim, n_channels)"""
        # Initialization
        X = np.empty((self.batch_size, *self.img_shape))

        # Generate data
        for i, file in enumerate(filenames):
            img = Image.open(path.join(self.data_folder, file))
            # crop and store sample
            data = self.__crop(np.asarray(img, dtype="uint8"))

            X[i, ] = data

        return X
