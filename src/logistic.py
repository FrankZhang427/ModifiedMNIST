#!/usr/bin/env python
"""
Logistic Regression Classifier for Modified MNIST
using Scikit-Learn LogisticRegression
"""
__author__ = "Zhiguo Zhang 260550226"

import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import numpy as np
from PIL import Image
from sklearn.preprocessing import StandardScaler
from sklearn import metrics

_NUM_INSTANCE_ = 50000
_NUM_TEST_ = 10000
_VAL_SET_ = 5000
_SIZE_ = 64
_RESULT_SIZE_ = 40


def main():
    scaler = StandardScaler()
    try:
        train_x = pickle.load(open("images_train.p", "rb"))
        val_x = pickle.load(open("images_val.p", "rb"))
        test_x = pickle.load(open("images_test.p", "rb"))
        train_y = pickle.load(open("labels_train.p", "rb"))
        val_y = pickle.load(open("labels_val.p", "rb"))
        print("Pickles loaded from disk")
    except:
        train_x, train_y, val_x, val_y, test_x = load_dataset()
        pickle.dump(train_y, open("labels_train.p", "wb"))
        pickle.dump(val_y, open("labels_val.p", "wb"))
        print("labels pickled")
        pickle.dump(train_x, open("images_train.p", "wb"))
        print("training images pickled")
        pickle.dump(val_x, open("images_val.p", "wb"))
        print("validation images pickled")
        pickle.dump(test_x, open("images_test.p", "wb"))
        print("test images pickled")

    train_x = img_to_array(data=train_x, num_instance=(_NUM_INSTANCE_ - _VAL_SET_))
    train_x = scaler.fit_transform(train_x)

    val_x = img_to_array(data=val_x, num_instance=_VAL_SET_)
    val_x = scaler.transform(val_x)

    test_x = img_to_array(data=test_x, num_instance=_NUM_TEST_)
    test_x = scaler.transform(test_x)

    print("Data Loaded")
    lr_version = "lr_" + str(_VAL_SET_) + ".p"
    try:
        lr_classifier = pickle.load(open(lr_version, "rb"))
        print("Logistic Regression Classifier pickles loaded from disk, Version:", lr_version)
    except pickle.PickleError:
        lr_classifier = LogisticRegression(penalty='l2', solver='sag', tol=0.01, n_jobs=-1, verbose=1,
                                           multi_class='multinomial')
        print("Classifier initialized")
        lr_classifier.fit(train_x, train_y.ravel())
        print("Classifier fitted")
        pickle.dump(lr_classifier, open(lr_version, "wb"))
        print("Logistic Regression Classifier Pickled, Version:", lr_version)

    prediction = lr_classifier.predict(val_x)
    print(classification_report(val_y, prediction))
    print(metrics.accuracy_score(val_y, prediction))

    prediction = lr_classifier.predict(test_x)
    lr_fname = "lr_out_" + str(_VAL_SET_) + ".csv"
    with open(lr_fname, "w") as f_out:
        f_out.write("Id,Label\n")
        for i in range(len(prediction)):
            f_out.write("%d,%d\n" % (i + 1, prediction[i]))
    print("File Written")


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
    train_x, val_x = train_x[:-5000], train_x[-5000:]
    train_y, val_y = train_y[:-5000], train_y[-5000:]
    return train_x, train_y, val_x, val_y, test_x


"""
    Convert image matrices into a flat array for logistic regression
        data:           - matrix to be converted
        num_instance:   - size of data to be converted
        img_size:       - image size. Default: 64
        return:         - array-like data
"""


def img_to_array(data, num_instance, img_size=64):
    out = np.zeros((num_instance, img_size * img_size))
    for i in range(num_instance):
        img = np.asanyarray(Image.fromarray(data[i])).flatten()
        out[i, :] = img
    return out


if __name__ == "__main__":
    main()
