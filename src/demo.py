import matplotlib

__author__ = "Zhiguo Zhang 260550226"

import Augmentor
import numpy as np
from PIL import Image
import tensorflow as tf
import pickle
from skimage.util import random_noise
import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter

import csv
import pickle
import sys

import keras
import numpy as np
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.layers import Dense, Dropout, Flatten, Lambda
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight

import numpy as np
import matplotlib as mp
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.examples.tutorials.mnist import input_data
import math

def elastic_transform(image, alpha, sigma, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_.


    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognition, 2003.
    """
    assert len(image.shape) == 2

    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha

    x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
    indices = np.reshape(x + dx, (-1, 1)), np.reshape(y + dy, (-1, 1))

    return map_coordinates(image, indices, order=1).reshape(shape)


if __name__ == '__main__':
    # x = np.loadtxt("toy_img.csv", delimiter=",")  # load from text
    # x = x.reshape(-1, 64, 64)  # reshape
    # x = np.uint8(x)
    train_x = pickle.load(open("images_train.p", "rb"))
    print(train_x.shape)
    # val_x = pickle.load(open("images_val.p", "rb"))
    test_x = pickle.load(open("images_test.p", "rb"))
    train_y = pickle.load(open("labels_train.p", "rb"))
    # val_y = pickle.load(open("labels_val.p", "rb"))
    print("Pickles loaded from disk")
    # img = cv2.imread('Samples/train_0.png')
    img = train_x[0]
    print(img)
    print(img.shape)
    # blur = cv2.blur(img, (5, 5))

    distort = elastic_transform(image=img, alpha=50, sigma=5, random_state=np.random.RandomState(12345))
    print(distort)
    plt.subplot(121), plt.imshow(img, cmap='gray'), plt.title('Original')
    plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(distort, cmap='gray'), plt.title('Elastic Distorted')
    plt.xticks([]), plt.yticks([])
    plt.show()
    # img = train_x[0]
    # noise_img = random_noise(img, mode='gaussian', seed=None, clip=True)
    # img = Image.fromarray(noise_img, 'L')
    # img.save('train_noise_1.png')
    # for i in range(20):
    #     img = Image.fromarray(train_x[i], 'L')
    #     img_name = 'train_' + str(i) +'.png'
    #     img.save(img_name)
    #     img = Image.fromarray(val_x[i], 'L')
    #     img_name = 'val_' + str(i) +'.png'
    #     img.save(img_name)
    #     img = Image.fromarray(test_x[i], 'L')
    #     img_name = 'test_' + str(i) +'.png'
    #     img.save(img_name)
        # img.show()
    # mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    # train_data = mnist.train.images  # Returns np.array
    # train_data = train_data.reshape([-1, 28, 28])
    # train_data = train_data * 255
    # train_data = train_data.astype('uint8')
    # print(train_data.shape)
    # train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
    # eval_data = mnist.test.images  # Returns np.array
    # eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)
    # for i in range(20):
    #     img = Image.fromarray(train_data[i], 'L')
    #     img_name = 'mnist_train_' + str(i) +'.png'
    #     img.save(img_name)
    #
    # with open("first_img.csv", "w") as f_out:
    #     for i in range(64):
    #         for j in range(64):
    #             if train_x[0,i,j] == 255:
    #                 f_out.write("%d," % train_x[0,i,j])
    #             else:
    #                 f_out.write("0,")
    #         f_out.write("\n")
    # print("File Written")
    # img = np.zeros([64, 64], dtype='uint8')
    # for i in range(64):
    #     for j in range(64):
    #         if train_x[0, i, j] == 255:
    #             img[i, j] = 255
    #         else:
    #             img[i, j] = 0
    # img = Image.fromarray(img, 'L')
    # img.save('first_img_train.png')
    # try:
    #     x_train = pickle.load(open("images_train.p", "rb"))
    #     x_predict = pickle.load(open("images_test.p", "rb"))
    #     y_train = pickle.load(open("labels_train.p", "rb"))
    #     print("Pickles loaded from disk")
    # except:
    #     x_train, y_train, x_predict = load_dataset()
    #     pickle.dump(y_train, open("labels_train.p", "wb"))
    #     print("labels pickled")
    #     pickle.dump(x_train, open("images_train.p", "wb"))
    #     print("training images pickled")
    #     pickle.dump(x_predict, open("images_test.p", "wb"))
    #     print("test images pickled")
    #
    # batch_size = 256
    # num_classes = 40
    # epochs = 50
    # aug_epochs = 100
    # test_size = 0.05
    # img_rows, img_cols = 64, 64
    #
    # print(x_train.shape)
    # input_shape = (img_rows, img_cols, 1)
    #
    # def norm_input(x):
    #     return x
    #
    # model = Sequential([
    #     Lambda(norm_input, input_shape=input_shape),
    #
    #     Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same', input_shape=input_shape),
    #     Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'),
    #     BatchNormalization(),
    #     MaxPooling2D(pool_size=(2, 2)),
    #     Dropout(0.25),
    #
    #     Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'),
    #     Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'),
    #     BatchNormalization(),
    #     MaxPooling2D(pool_size=(2, 2)),
    #     Dropout(0.25),
    #
    #     Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'),
    #     Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'),
    #     Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'),
    #     BatchNormalization(),
    #     MaxPooling2D(pool_size=(2, 2)),
    #     Dropout(0.25),
    #
    #     # Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same'),
    #     # Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same'),
    #     # Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same'),
    #     # BatchNormalization(),
    #     # MaxPooling2D(pool_size=(2, 2)),
    #     # Dropout(0.25),
    #
    #     Flatten(),
    #
    #     Dense(512, activation='relu'),
    #     BatchNormalization(),
    #     Dropout(0.5),
    #     Dense(512, activation='relu'),
    #     BatchNormalization(),
    #     Dropout(0.5),
    #     Dense(512, activation='relu'),
    #     BatchNormalization(),
    #     Dropout(0.5),
    #     # Dense(512, activation='relu'),
    #     # BatchNormalization(),
    #     # Dropout(0.5),
    #     Dense(40, activation='softmax')
    # ])
    #
    # model.compile(loss=keras.losses.categorical_crossentropy,
    #               optimizer=keras.optimizers.Adam(lr=0.001),
    #               metrics=['accuracy'])
    # model.summary()

    p = Augmentor.Pipeline("./Samples/train_0")
    p.skew(probability=1, magnitude=0.6)
    p.status()
    distort = p.sample(1)
    print(type(distort))
    fname = 'Samples/train_0/output/train_0_439ac30d-ccde-4d08-bd62-95ced4ed4792.JPEG'
    image = Image.open(fname).convert("L")
    arr = np.asarray(image)
    # g = p.keras_generator_from_array(train_x, train_y, batch_size=10)
    # X, y = next(g)
    # print(X.shape)
    # for i in range(10):
    #     img = Image.fromarray(X[i].reshape(64, 64))
    #     print(y[i])
    #     img.save('train_noise_augmentor_'+str(i)+'.png')
    plt.subplot(121), plt.imshow(img, cmap='gray'), plt.title('Original')
    plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(arr, cmap='gray'), plt.title('Perspective Skew')
    plt.xticks([]), plt.yticks([])
    plt.show()

    tf.reset_default_graph()

    x = tf.placeholder(tf.float32, [None, 4096], name="x-in")
    true_y = tf.placeholder(tf.float32, [None, 40], name="y-in")
    keep_prob = tf.placeholder("float")
    sess = tf.Session()

    imageToUse = img.reshape([-1, 4096])
    plt.imshow(np.reshape(imageToUse, [64, 64]), interpolation="nearest", cmap="gray")

    x_image = tf.reshape(x, [-1, 64, 64, 1])
    hidden_1 = slim.conv2d(x_image, 32, [3, 3])
    # pool_1 = slim.max_pool2d(hidden_1, [2, 2])
    hidden_2 = slim.conv2d(hidden_1, 32, [3, 3])
    # pool_1 = slim.max_pool2d(hidden_2, [2, 2])

    hidden_3 = slim.conv2d(hidden_2, 64, [3, 3])
    # pool_1 = slim.max_pool2d(hidden_1, [2, 2])
    hidden_4 = slim.conv2d(hidden_3, 64, [3, 3])
    # pool_2 = slim.max_pool2d(hidden_2, [2, 2])

    hidden_5 = slim.conv2d(hidden_4, 128, [3, 3])
    # pool_1 = slim.max_pool2d(hidden_1, [2, 2])
    hidden_6 = slim.conv2d(hidden_5, 128, [3, 3])
    hidden_7 = slim.conv2d(hidden_6, 128, [3, 3])
    # pool_3 = slim.max_pool2d(hidden_7, [2, 2])

    hidden_8 = slim.dropout(hidden_7, keep_prob)
    out_y = slim.fully_connected(slim.flatten(hidden_8), 40, activation_fn=tf.nn.softmax)

    cross_entropy = -tf.reduce_sum(true_y * tf.log(out_y))
    correct_prediction = tf.equal(tf.argmax(out_y, 1), tf.argmax(true_y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    batchSize = 50
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    # for i in range(1001):
    #     batch = mnist.train.next_batch(batchSize)
    #     sess.run(train_step, feed_dict={x: batch[0], true_y: batch[1], keep_prob: 0.5})
    #     if i % 100 == 0 and i != 0:
    #         trainAccuracy = sess.run(accuracy, feed_dict={x: batch[0], true_y: batch[1], keep_prob: 1.0})
    #         print("step %d, training accuracy %g" % (i, trainAccuracy))

    def getActivations(layer, stimuli, shapes):
        units = sess.run(layer, feed_dict={x: np.reshape(stimuli, [-1, shapes], order='F'), keep_prob: 1.0})
        plotNNFilter(units)
        return units


    def plotNNFilter(units):
        filters = units.shape[3]
        print(filters)
        plt.figure(1, figsize=(64, 64))
        n_columns = 6
        n_rows = math.ceil(filters / n_columns) + 1
        # for i in range(filters):
            # plt.subplot(n_rows, n_columns, i + 1)
            # plt.title('Filter ' + str(i))
        plt.imshow(units[0, :, :, 0], interpolation="nearest", cmap="gray")
        plt.show()

    imageToUse = getActivations(hidden_1, imageToUse, 4096)
    print(imageToUse.shape)
    imageToUse = getActivations(hidden_2, imageToUse[0, :, :, 0], 4096)
    print(imageToUse.shape)
    imageToUse = getActivations(hidden_3, imageToUse[0, :, :, 0], 4096)
    print(imageToUse.shape)
    imageToUse = getActivations(hidden_4, imageToUse[0, :, :, 0], 4096)
    print(imageToUse.shape)
    imageToUse = getActivations(hidden_5, imageToUse[0, :, :, 0], 4096)
    print(imageToUse.shape)
    imageToUse = getActivations(hidden_6, imageToUse[0, :, :, 0], 4096)
    print(imageToUse.shape)
    imageToUse = getActivations(hidden_7, imageToUse[0, :, :, 0], 4096)
    print(imageToUse.shape)



