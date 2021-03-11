"""
Helper Functions for COMP 551: Project 3
"""
__author__ = "Zhiguo Zhang 260550226"

import numpy as np
from PIL import Image

"""
    General Loader for all classifiers with functionality to generate validation set
        val_set:        - the size of validation set, which will be sampled from the last of input training images
        num_instance:   - number of instances to be loaded from training images. Default: -1 (all images)
        img_size:       - the size of image. Default: 64
        tr_x:           - file name of training images
        tr_y:           - file name of training labels
        t_x:            - file name of test images
        return:         - training set(with labels), validation set(with labels), test set(without labels)
"""


def load_dataset(val_set, num_instance=-1, img_size=64, tr_x='train_x.csv', tr_y='train_y.csv', t_x='test_x.csv'):
    def load_data_images(filename, num_inst=-1):
        data = np.loadtxt(filename, delimiter=",")  # load from text
        data = data.reshape(num_inst, img_size, img_size)  # reshape
        data = np.uint8(data)
        return data

    def load_data_labels(filename, num_inst=-1):
        data = np.loadtxt(filename, delimiter=",")  # load from text
        data = data.reshape(num_inst, 1)  # reshape
        data = np.uint8(data)
        return data

    train_x = load_data_images(tr_x, num_inst=num_instance)
    print(tr_x, " loaded")
    train_y = load_data_labels(tr_y, num_inst=num_instance)
    print(tr_y, " loaded")
    test_x = load_data_images(t_x, num_inst=-1)
    print(t_x, " loaded")
    train_x, val_x, train_y, val_y = train_x[:-val_set], train_x[-val_set:], train_y[:-val_set], train_y[-val_set:]
    return train_x, train_y, val_x, val_y, test_x


"""
    Convert image matrices into a flat array for logistic regression
        data:           - matrix to be converted
        num_instance:   - size of data to be converted
        img_size:       - image size. Default: 64
        return:         - array-like data
"""


def img_to_array(data, num_instance, img_size=64):
    out = np.zeros((img_size*img_size, num_instance))
    for i in range(num_instance):
        img = np.asanyarray(Image.fromarray(data[i])).flatten()
        out[i, :] = img
    return out
