#!/usr/bin/env python
"""
Modified MNIST image classification implementation based on Keras
Readings:
http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.160.8494&rep=rep1&type=pdf
https://arxiv.org/pdf/1202.2745.pdf
http://augmentor.readthedocs.io/en/master/userguide/mainfeatures.html
https://adeshpande3.github.io/adeshpande3.github.io/The-9-Deep-Learning-Papers-You-Need-To-Know-About.html
https://www.cs.umb.edu/~twang/file/tricks_from_dl.pdf

Not used ideas:
https://www.cv-foundation.org/openaccess/content_cvpr_2015/app/2B_004.pdf

R-CNN(Regional)
https://github.com/rbgirshick/rcnn
https://blog.athelas.com/a-brief-history-of-cnns-in-image-segmentation-from-r-cnn-to-mask-r-cnn-34ea83205de4
https://arxiv.org/pdf/1708.02002.pdf
https://arxiv.org/pdf/1311.2524.pdf
https://arxiv.org/pdf/1506.01497.pdf
https://arxiv.org/pdf/1504.08083.pdf
https://arxiv.org/pdf/1703.06870.pdf
https://courses.cs.washington.edu/courses/cse590v/14au/cse590v_wk1_rcnn.pdf

Residual learning:
https://arxiv.org/pdf/1512.03385v1.pdf
"""

from __future__ import print_function

__author__ = "Zhiguo Zhang 260550226"

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
from keras.utils import plot_model
from matplotlib import pyplot as plt

np.random.seed(12345)


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


def load_dataset(num_instance=-1, img_size=64, tr_x='train_x.csv', tr_y='train_y.csv', t_x='test_x.csv'):
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
    return train_x, train_y, test_x


"""
    Mapping labels to [0,39]
     [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
     20, 21, 24, 25, 27, 28, 30, 32, 35, 36, 40, 42, 45, 48, 49, 54, 56, 63, 64, 72, 81]
"""


def label_map(in_labels, num_instance):
    out_labels = np.zeros((in_labels.shape[0],))
    for i in range(num_instance):
        if in_labels[i] < 19:
            out_labels[i] = in_labels[i]
        if in_labels[i] == 20:
            out_labels[i] = 19
        if in_labels[i] == 21:
            out_labels[i] = 20
        if in_labels[i] == 24:
            out_labels[i] = 21
        if in_labels[i] == 25:
            out_labels[i] = 22
        if in_labels[i] == 27:
            out_labels[i] = 23
        if in_labels[i] == 28:
            out_labels[i] = 24
        if in_labels[i] == 30:
            out_labels[i] = 25
        if in_labels[i] == 32:
            out_labels[i] = 26
        if in_labels[i] == 35:
            out_labels[i] = 27
        if in_labels[i] == 36:
            out_labels[i] = 28
        if in_labels[i] == 40:
            out_labels[i] = 29
        if in_labels[i] == 42:
            out_labels[i] = 30
        if in_labels[i] == 45:
            out_labels[i] = 31
        if in_labels[i] == 48:
            out_labels[i] = 32
        if in_labels[i] == 49:
            out_labels[i] = 33
        if in_labels[i] == 54:
            out_labels[i] = 34
        if in_labels[i] == 56:
            out_labels[i] = 35
        if in_labels[i] == 63:
            out_labels[i] = 36
        if in_labels[i] == 64:
            out_labels[i] = 37
        if in_labels[i] == 72:
            out_labels[i] = 38
        if in_labels[i] == 81:
            out_labels[i] = 39
    return out_labels


def label_rev_map(in_labels, num_instance):
    out_labels = np.zeros((in_labels.shape[0],))
    for i in range(num_instance):
        if in_labels[i] < 19:
            out_labels[i] = in_labels[i]
        if in_labels[i] == 19:
            out_labels[i] = 20
        if in_labels[i] == 20:
            out_labels[i] = 21
        if in_labels[i] == 21:
            out_labels[i] = 24
        if in_labels[i] == 22:
            out_labels[i] = 25
        if in_labels[i] == 23:
            out_labels[i] = 27
        if in_labels[i] == 24:
            out_labels[i] = 28
        if in_labels[i] == 25:
            out_labels[i] = 30
        if in_labels[i] == 26:
            out_labels[i] = 32
        if in_labels[i] == 27:
            out_labels[i] = 35
        if in_labels[i] == 28:
            out_labels[i] = 36
        if in_labels[i] == 29:
            out_labels[i] = 40
        if in_labels[i] == 30:
            out_labels[i] = 42
        if in_labels[i] == 31:
            out_labels[i] = 45
        if in_labels[i] == 32:
            out_labels[i] = 48
        if in_labels[i] == 33:
            out_labels[i] = 49
        if in_labels[i] == 34:
            out_labels[i] = 54
        if in_labels[i] == 35:
            out_labels[i] = 56
        if in_labels[i] == 36:
            out_labels[i] = 63
        if in_labels[i] == 37:
            out_labels[i] = 64
        if in_labels[i] == 38:
            out_labels[i] = 72
        if in_labels[i] == 39:
            out_labels[i] = 81
    return out_labels


def main(version, batch_size=64, epochs=20, learning_rate=0.001):
    try:
        x_train = pickle.load(open("images_train.p", "rb"))
        x_predict = pickle.load(open("images_test.p", "rb"))
        y_train = pickle.load(open("labels_train.p", "rb"))
        print("Pickles loaded from disk")
    except:
        x_train, y_train, x_predict = load_dataset()
        pickle.dump(y_train, open("labels_train.p", "wb"))
        print("labels pickled")
        pickle.dump(x_train, open("images_train.p", "wb"))
        print("training images pickled")
        pickle.dump(x_predict, open("images_test.p", "wb"))
        print("test images pickled")

    # batch_size = 64
    num_classes = 40
    # epochs = 20
    aug_epochs = 1
    test_size = 0.1
    img_rows, img_cols = 64, 64

    print(x_train.shape)
    input_shape = (img_rows, img_cols, 1)

    ###############################################################################
    distort_1 = np.zeros(shape=x_train.shape, dtype='float32')
    distort_2 = np.zeros(shape=x_train.shape, dtype='float32')
    distort_3 = np.zeros(shape=x_train.shape, dtype='float32')

    for i in range(x_train.shape[0]):
        print(i)
        distort_1[i] = elastic_transform(image=x_train[i], alpha=36, sigma=10)
        distort_2[i] = elastic_transform(image=x_train[i], alpha=36, sigma=8)
        distort_3[i] = elastic_transform(image=x_train[i], alpha=36, sigma=6)

    x_train = np.concatenate((x_train, distort_1, distort_2, distort_3), axis=0)
    y_train = np.concatenate((y_train, y_train, y_train, y_train), axis=0)
    ###############################################################################

    x_train = x_train.reshape((-1, img_rows, img_cols, 1))
    x_predict = x_predict.reshape((-1, img_rows, img_cols, 1))
    print('y_train shape:', y_train.shape)  # (-1,)
    y_train = label_map(y_train, y_train.shape[0])
    print('y_train shape:', y_train.shape)
    y_train = y_train.reshape((y_train.shape[0],))
    x_train = x_train.astype('float32')
    x_predict = x_predict.astype('float32')
    x_train /= 255
    x_predict /= 255
    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=test_size, random_state=12345)

    print('x_train shape:', x_train.shape)  # (-1,28,28,1)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')
    print('y_train shape:', y_train.shape)  # (-1,)
    class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
    print(class_weights)

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    mean_px = x_train.mean().astype(np.float32)
    std_px = x_train.std().astype(np.float32)

    def norm_input(x):
        return (x - mean_px) / std_px

    model = Sequential([
        Lambda(norm_input, input_shape=input_shape),

        Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same', input_shape=input_shape),
        Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'),
        Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'),
        Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'),
        Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        # Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same'),
        # Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same'),
        # Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same'),
        # BatchNormalization(),
        # MaxPooling2D(pool_size=(2, 2)),
        # Dropout(0.25),

        Flatten(),

        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        # Dense(512, activation='relu'),
        # BatchNormalization(),
        # Dropout(0.5),
        # Dense(512, activation='relu'),
        # BatchNormalization(),
        # Dropout(0.5),
        Dense(40, activation='softmax')
    ])

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(lr=learning_rate),
                  metrics=['accuracy'])
    # plot_model(model, to_file='model.png')
    original = sys.stdout
    sys.stdout = open('history_' + str(version) + '.txt', 'w')

    model.summary()
    # Training
    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        validation_data=(x_test, y_test),
                        class_weight=class_weights)
    # Changing learning rate
    # model.optimizer.lr = 0.0001
    # model.fit(x_train, y_train,
    #           batch_size=batch_size,
    #           epochs=epochs,
    #           verbose=1,
    #           validation_data=(x_test, y_test),
    #           class_weight=class_weights)
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    print(history.history.keys())
    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('model_' + str(version) + 'acc.png')
    plt.close()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('model_' + str(version) + 'loss.png')
    plt.close()
    gen = ImageDataGenerator(rotation_range=15, width_shift_range=0.1, shear_range=0.3,
                             height_shift_range=0.1, zoom_range=0.08)  # changed from 8, 0.08, 0.3, 0.08
    batches = gen.flow(x_train, y_train, batch_size=batch_size)
    val_batches = gen.flow(x_test, y_test, batch_size=batch_size)
    model.fit_generator(batches, steps_per_epoch=x_train.shape[0] // batch_size, epochs=aug_epochs,
                        validation_data=val_batches, validation_steps=x_test.shape[0] // batch_size,
                        use_multiprocessing=False, class_weight=class_weights)

    sys.stdout = original

    # Saving model
    model_json = model.to_json()
    json_name = "model_" + str(version) + ".json"
    h5_name = "model_" + str(version) + ".h5"
    with open(json_name, "w") as json_file:
        json_file.write(model_json)
    model.save_weights(h5_name)
    print("Saved model to disk")

    # Prediction
    y_predict = model.predict_classes(x_predict, batch_size=batch_size, verbose=0)
    y_predict = label_rev_map(y_predict, y_predict.shape[0])

    # Write prediction to file
    pred_name = "model_" + str(version) + "_prediction.csv"
    with open(pred_name, 'w', newline='') as f:
        fieldnames = ["Id", "Label"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(len(y_predict)):
            writer.writerow({'Id': i + 1, 'Label': np.uint8(y_predict[i])})
    print("File Written")


if __name__ == '__main__':
    main(100)
    # main(9, batch_size=64)
    # main(10, batch_size=128)
    # main(11, batch_size=256)
    #
    # main(12, batch_size=256, learning_rate=0.1)
    # main(13, batch_size=256, learning_rate=0.01)
    # main(14, batch_size=256, learning_rate=0.001) # epochs = 20
    # main(15, batch_size=256, learning_rate=0.0001)
    #
    # main(16, batch_size=256, learning_rate=0.001, epochs=50)
    # main(17, batch_size=256, learning_rate=0.001, epochs=100)
    # main(18, batch_size=256, learning_rate=0.001, epochs=200)
