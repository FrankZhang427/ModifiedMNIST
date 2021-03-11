# README #

### What is this repository for? ###

Project	#3:		Modified digits
Instructions:
http://cs.mcgill.ca/~jpineau/comp551/Project3_instructions.pdf

### Logistic Regression ###

"logistic.py" is a logistic classifier implemented using Sklearn package.
To run "logistic.py", you should put the "train_x.csv", "train_y.csv", "test_x.csv" in the same directory as "logistic.py".

"logistic.py" works on python 3.6.x

### Neural Network ###

Version 1:
"NN_3_layer.py" and "NN_4_layer.py" train a 3-layer fully connected neuron network and a 4-layer network respectively,
with tanh as the activation function and the softmax function implemented at the last stage.
A weighting function is incorporated in the gradient descent process to deal with imbalanced training classes.
Different hyper-parameters can be specified at the beginning of the two files,
and accuracies and unweighted losses will be output during training.

Version 2:
"working on total.ipynb" is the script where the neural network class is defined and the training process take place. You should run the "preprocessing_data" first and then run this file.
"preprocessing_data.ipynb" is the script which load the original train_x.csv, train_y.csv, test_x.csv and preprocess it. Then it saves the data by pickle package as lists of integers.
To run these file, you should put the "train_x.csv", "train_y.csv", "test_x.csv" in the same directory as these two files.
Run "preprocessing_data.ipynb" first. and then Run "working on total.ipynb" to training the neural network and get the test result.
The test result is saved in file "test_output" by default unless you pass the filename to the function "nn.test()"

Both version works on python 2.7.x

### Convolutional Neural Network ###

cnn_keras.py is CNN test code for hyper-parameters selection implemented using Keras package.
cnn_keras_augmentor.py is CNN with Augmentor added for data augmentation. This is the final model for Kaggle competition.
To run above scripts, you should put the "train_x.csv", "train_y.csv", "test_x.csv" in the same directory.

Both scripts work on python 3.6.x