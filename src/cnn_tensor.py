#!/usr/bin/env python
"""
Convolutional Neural Network(CNN) built on tensorflow
"""
__author__ = "Zhiguo Zhang 260550226"

import pickle
import tensorflow as tf
import numpy as np
from PIL import Image

_NUM_INSTANCE_ = 50000
_NUM_TEST_ = 10000
_VAL_SET_ = 5000
_SIZE_ = 64  # 64
_RESULT_SIZE_ = 40  # 40
tf.logging.set_verbosity(tf.logging.INFO)
version = 8


# adapted from https://www.tensorflow.org/tutorials/layers

def cnn_model_fn(features, labels, mode):
  """Model function for CNN."""
  # Input Layer
  # Reshape X to 4-D tensor: [batch_size, width, height, channels]
  # MNIST images are 28x28 pixels, and have one color channel
  input_layer = tf.reshape(features["x"], [-1, 64, 64, 1])

  # Convolutional Layer #1
  # Computes 32 features using a 5x5 filter with ReLU activation.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 28, 28, 1]
  # Output Tensor Shape: [batch_size, 28, 28, 32]
  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=32,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #1
  # First max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 28, 28, 32]
  # Output Tensor Shape: [batch_size, 14, 14, 32]
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

  # Convolutional Layer #2
  # Computes 64 features using a 5x5 filter.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 14, 14, 32]
  # Output Tensor Shape: [batch_size, 14, 14, 64]
  conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=64,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #2
  # Second max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 14, 14, 64]
  # Output Tensor Shape: [batch_size, 7, 7, 64]
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

  # Flatten tensor into a batch of vectors
  # Input Tensor Shape: [batch_size, 7, 7, 64]
  # Output Tensor Shape: [batch_size, 7 * 7 * 64]
  pool2_flat = tf.reshape(pool2, [-1, 16 * 16 * 64])

  # Dense Layer
  # Densely connected layer with 1024 neurons
  # Input Tensor Shape: [batch_size, 7 * 7 * 64]
  # Output Tensor Shape: [batch_size, 1024]
  dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)

  # Add dropout operation; 0.6 probability that element will be kept
  dropout = tf.layers.dropout(
      inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

  # Logits layer
  # Input Tensor Shape: [batch_size, 1024]
  # Output Tensor Shape: [batch_size, 10]
  logits = tf.layers.dense(inputs=dropout, units=40)

  predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
  }
  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Calculate Loss (for both TRAIN and EVAL modes)
  onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=40)
  loss = tf.losses.softmax_cross_entropy(
      onehot_labels=onehot_labels, logits=logits)

  # Configure the Training Op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0001)
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  # Add evaluation metrics (for EVAL mode)
  eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(
          labels=labels, predictions=predictions["classes"])}
  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def cnn_model_fn2(features, labels, mode):
    # Input layer
    input_layer = tf.reshape(features["x"], [-1, _SIZE_, _SIZE_, 1])

    # convolution layer activated with ReLU activation function for nonlinearity
    conv1 = tf.layers.conv2d(inputs=input_layer, filters=64, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)
    conv2 = tf.layers.conv2d(inputs=conv1, filters=64, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)
    pool1 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
    conv3 = tf.layers.conv2d(inputs=pool1, filters=64, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)
    conv4 = tf.layers.conv2d(inputs=conv3, filters=64, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2], strides=2)
    conv5 = tf.layers.conv2d(inputs=pool2, filters=64, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)
    conv6 = tf.layers.conv2d(inputs=conv5, filters=64, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)
    # pool3 = tf.layers.max_pooling2d(inputs=conv6, pool_size=[2, 2], strides=2)

    # Dense layers and logits layers
    pool3_flat = tf.reshape(conv6, [-1, _SIZE_ * _SIZE_ * 4])  # _SIZE_ * _SIZE_ * 64 / 64
    hidden1 = tf.layers.dense(inputs=pool3_flat, units=4096, activation=tf.nn.relu)
    dropout1 = tf.layers.dropout(inputs=hidden1, rate=0.2, training=mode == tf.estimator.ModeKeys.TRAIN)
    hidden2 = tf.layers.dense(inputs=dropout1, units=4096, activation=tf.nn.relu)
    dropout2 = tf.layers.dropout(inputs=hidden2, rate=0.2, training=mode == tf.estimator.ModeKeys.TRAIN)
    logits = tf.layers.dense(inputs=dropout2, units=_RESULT_SIZE_)
    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=_RESULT_SIZE_)
    loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        # optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)  # Can use adam here
        optimizer = tf.train.MomentumOptimizer(momentum=0.9, learning_rate=0.1, use_nesterov=True) # Can use adam here
        train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {"accuracy": tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


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
    return out.astype('float32')


def main(unused_argv):
    try:
        train_x = pickle.load(open("images_train.p", "rb"))
        val_x = pickle.load(open("images_val.p", "rb"))
        test_x = pickle.load(open("images_test.p", "rb"))
        train_y = pickle.load(open("labels_train.p", "rb"))
        val_y = pickle.load(open("labels_val.p", "rb"))
        print("Pickles loaded from disk")
    except pickle.PickleError:
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

    # train_x = train_x.astype('float32') / 255.0
    train_x = (train_x == train_x.max(axis=1, keepdims=True)).astype(int)
    train_data = img_to_array(data=train_x, num_instance=(_NUM_INSTANCE_ - _VAL_SET_))
    print(train_x.shape)
    print(train_x.dtype)
    # val_x = val_x.astype('float32') / 255.0
    val_x = (val_x == val_x.max(axis=1, keepdims=True)).astype(int)
    eval_data = img_to_array(data=val_x, num_instance=_VAL_SET_)
    print(val_x.shape)
    print(val_x.dtype)
    # test_x = test_x.astype('float32') / 255.0
    test_x = (test_x == test_x.max(axis=1, keepdims=True)).astype(int)
    test_x = img_to_array(data=test_x, num_instance=_NUM_TEST_)
    print(test_x.shape)
    print(test_x.dtype)
    train_labels = train_y.reshape((-1,)).astype('int32')
    print(train_y.shape)
    print(train_y.dtype)
    eval_labels = val_y.reshape((-1,)).astype('int32')
    print(val_y.shape)
    print(val_y.dtype)
    # Creating Estimator
    mnist_classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn2, model_dir="/tmp/convnet_model")

    # Set up logging for predictions
    # Log the values in the "Softmax" tensor with label "probabilities"
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=50)

    # Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data},
        y=train_labels,
        batch_size=100,
        num_epochs=None,
        shuffle=True)
    mnist_classifier.train(input_fn=train_input_fn, steps=100000)  # hooks=[logging_hook]

    # Evaluate the model and print results
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data},
        y=eval_labels,
        num_epochs=1,
        shuffle=False)
    eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)

    # Prediction on new images
    predict_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x": test_x}, num_epochs=1, shuffle=False)
    predictions = mnist_classifier.predict(input_fn=predict_input_fn)
    print(type(predictions))   # <class 'generator'>
    cnn_fname = "cnn_out_" + str(version) + ".csv"
    index = 1
    with open(cnn_fname, "w") as f_out:
        f_out.write("Id,Label\n")
        for pred in predictions:
            f_out.write("%d,%d\n" % (index, pred['classes']))
            index = index + 1
    print("File Written")


########################################################################################################################
# original main() function in tutorial
########################################################################################################################
def main2(unused_argv):
    # Load training and eval data
    mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    train_data = mnist.train.images  # Returns np.array
    print(train_data.shape)
    print(train_data.dtype)
    # n_train, n_pixel = train_data.shape
    # for i in range(n_pixel):
    #     print(train_data[0, i])
    train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
    print(train_labels.shape)
    print(train_labels.dtype)
    eval_data = mnist.test.images  # Returns np.array
    print(eval_data.shape)
    print(eval_data.dtype)
    eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)
    print(eval_labels.shape)
    print(eval_labels.dtype)
    # Create the Estimator
    mnist_classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn2, model_dir="/tmp/mnist_convnet_model")

    # Set up logging for predictions
    # Log the values in the "Softmax" tensor with label "probabilities"
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=50)

    # Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data},
        y=train_labels,
        batch_size=100,
        num_epochs=None,
        shuffle=True)
    mnist_classifier.train(input_fn=train_input_fn, steps=20000)  # hooks=[logging_hook]

    # Evaluate the model and print results
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data},
        y=eval_labels,
        num_epochs=1,
        shuffle=False)
    eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)


########################################################################################################################


if __name__ == '__main__':
    tf.app.run()
    # main(version=5)
    # main2()
