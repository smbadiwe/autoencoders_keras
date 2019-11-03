import numpy as np
from keras.utils import Sequence
from os import path
from PIL import Image
from math import floor, ceil
from keras.applications.vgg16 import preprocess_input


class DataGenerator(Sequence):
    """
    Data generator, built with COCO test2014 dataset in mind
    """
    def __init__(self, filenames, data_folder, batch_size=8, img_shape=(224, 224, 3),
                 shuffle=False):
        """
        Initialize data generator, built with COCO test2014 dataset in mind
        :param filenames:
        :param data_folder:
        :param batch_size:
        :param img_shape:
        :param shuffle:
        """
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
        if img.ndim == 2:
            img = np.broadcast_to(img[..., np.newaxis], (img.shape[0], img.shape[1], 3))
        w, h, _ = img.shape
        # try:
        #     w, h, _ = img.shape
        # except Exception:
        #     print(f"Img shape: {img.shape}")
        #     raise
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
            w, h = img.size
            if w < 224 or h < 224:
                continue
            try:
                # crop and store sample
                data = np.asarray(img, dtype="uint8")
                if data.ndim == 2:
                    continue
                if np.max(data) == 0:
                    print(f"file {file} HAS ALL ZEROs. Moving on")
                    continue
                data = self.__crop(data)
                X[i, ] = preprocess_input(data, mode="tf")
            except:
                print(f"file {file} failed. Moving on")
                continue
        return X
