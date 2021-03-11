# This file implements the 4-layer fully-connected feedforward neuron network. The activation function is tanh and softmax is implemented at the output layer. A weighting function is incorporated in the gradient descent process to deal with imbalanced training classes. Different hyper-parameters can be specified at the beginning of the two files, and accuracies and unweighted losses will be output during training. 
# NOTE: You may need to delete the comment on the forth line to execute the file.
# The struction of the network is essentially the same as that given in [1] and the code for the self-defined functions is based on that in [1].
# Reference: [1]. D. Britz, Implementing a Neural Network from Scratch in Python – An Introduction, 2015. [Online]. Available: http://www.wildml.com/2015/09/implementing-a-neural-network-from-scratch/. [Accessed: 12-Nov.-17].
#
# Author: Yunhua Zheng. Last modified 12 Nov, 2017.

# Parameters
# Input and output training file names
filename_y="train_y.csv"
filename_x="train_x.csv"

test_size=0.2 # The proportion of the test set to be used in validation

# Network and gradient descent parameters
input_dim = 4096 # input layer dimensionality
output_dim = 40 # output layer dimensionality
hdim1=128 # The number of nodes in hidden layer 1 (the second hidden layer)
hdim0_range=[256,512,1024] # The number of nodes in the first hidden layer
epsilon_range = [0.01,0.008,0.012] # Learning rate for gradient descent
num_passes_current=1000 # Number of updates to be performed
print_interval_current=50 # Interval for performance metrics' output


import numpy as np
# The three functions has been taken from [1] and modified to suit the 4-layer network. Besides, the third one has been modified to incorporate weighting of classes, to display different information, and to enable specification of more parameters when calling the function.
# 1. Function to evaluate the total loss on the dataset
def calculate_loss(model, x, y):
    num_examples=len(x)
    W0,b0,W1, b1, W2, b2 = model['W0'], model['b0'],model['W1'], model['b1'], model['W2'], model['b2']
    # Forward propagation to calculate our predictions
    z0 = x.dot(W0) + b0
    a0 = np.tanh(z0)
    z1 = a0.dot(W1) + b1
    a1 = np.tanh(z1)
    z2 = a1.dot(W2) + b2
    exp_scores = np.exp(z2)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    #** Calculating the loss
    corect_logprobs = -np.log(probs[range(num_examples), y])
    data_loss = np.sum(corect_logprobs) # 40 possible outcomes encoded using 1-hot encoding
    return 1./num_examples * data_loss

# 2. Function to predict the output (40 possible outcomes encoded using 1-hot encoding)
def predict(model, x):
    W0,b0,W1, b1, W2, b2 = model['W0'], model['b0'],model['W1'], model['b1'], model['W2'], model['b2']
    # Forward propagation to calculate our predictions
    z0 = x.dot(W0) + b0
    a0 = np.tanh(z0)
    z1 = a0.dot(W1) + b1
    a1 = np.tanh(z1)
    z2 = a1.dot(W2) + b2
    exp_scores = np.exp(z2)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    return np.argmax(probs, axis=1)

# 3. This function learns parameters for the neural network and returns the model.
# - hdim1: Number of nodes in the first hidden layer
# - num_passes: Number of passes through the training data for gradient descent
# - print_loss: If True, print the loss and accuracy every print_interval iterations
# - training: If training=True, both training set and validation set need to be input, i.e. X, Y, X_val and Y_val all need to be fed, and the validation error will be output as well print_loss=True.
def build_model(hdim0,hdim1, X, Y,training=False, X_val=None, Y_val=None, num_passes=2000, print_loss=False, print_interval=10):
    num_examples=len(train_x) # training set size
    #** Initialize the parameters to random values
    np.random.seed(45)
    W0 = np.random.randn(input_dim, hdim0) / np.sqrt(input_dim)
    b0 = np.zeros((1, hdim0))
    W1 = np.random.randn(hdim0, hdim1) / np.sqrt(hdim0)
    b1 = np.zeros((1, hdim1))
    W2 = np.random.randn(hdim1, output_dim) / np.sqrt(hdim1)
    b2 = np.zeros((1, output_dim))
    
    # This is what we return at the end
    model = {}
    loss=[]
    loss_val=[]
    accuracy=[]
    accuracy_val=[]
    
    # Weigh the classes such that misclassifying a less probable outcome gives a larger penalty (by weighing the parameters in obtaining learning rates)
    #from sklearn.utils import class_weight
    #class_weight = class_weight.compute_class_weight('balanced', np.unique(Y), Y)
    weights=np.zeros(40)
    classes=np.unique(Y)
    for i in range(len(classes)):
        weights[classes[i]]=2/float(list(Y).count(classes[i]))
    print "The weights are:"
    print weights

    # Gradient descent. For each batch...
    for i in range(0, num_passes):
    
        # Forward propagation
        z0 = X.dot(W0) # num_examples*hdim1
        a0 = np.tanh(z0) # num_examples*hdim1
        z1 = a0.dot(W1) # num_examples*hdim1
        a1 = np.tanh(z1) # num_examples*hdim1
        z2 = a1.dot(W2) # num_examples*output_dim
        exp_scores = np.exp(z2) # num_examples*output_dim
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True) # num_examples*output_dim
    
        # Backpropagation
        delta3 = probs # num_examples*output_dim
        delta3[range(num_examples), Y] -= 1
        temp=weights[Y] # num_examples*1
        delta3=delta3*temp[:, np.newaxis]
        dW2 = (a1.T).dot(delta3) # hdim1*output_dim
        db2 = np.sum(delta3, axis=0, keepdims=True) # 1*output_dim
        delta2 = delta3.dot(W2.T) * (1 - np.power(a1, 2)) # num_examples*hdim1
        dW1 = np.dot(a0.T, delta2) # input_dim*hdim1
        db1 = np.sum(delta2, axis=0) # 1*hdim1
        delta1 = delta2.dot(W1.T) * (1 - np.power(a0, 2))
        dW0 = np.dot(X.T, delta1) # input_dim*hdim0
        db0 = np.sum(delta1, axis=0) # 1*hdim0
        
        # Gradient descent parameter update
        W0 += -epsilon * dW0
        b0 += -epsilon * db0
        W1 += -epsilon * dW1
        b1 += -epsilon * db1
        W2 += -epsilon * dW2
        b2 += -epsilon * db2
        
        # Assign new parameters to the model
        model = { 'W0':W0,'b0':b0,'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}
        
        # Optionally print the loss
        if print_loss and (i+1) % print_interval == 0:
            loss_current=calculate_loss(model,X,Y)
            print "Loss after iteration %i on training set: %f" %(i, loss_current)
            loss.append(loss_current)
            
            # Calculate the achieved accuracy on the training set
            y_train=predict(model,X)
            accuracy.append(np.sum(Y==y_train)/float(len(Y)))
            print "Accuracy after iteration %i on training set: %f" %(i, accuracy[-1])
            
            if training==True:
                loss_current_val=calculate_loss(model,X_val,Y_val)
                print "Loss after iteration %i on validation set: %f" %(i, loss_current_val)
                loss_val.append(loss_current_val)
                # Calculate the achieved accuracy on the training set
                y_val=predict(model,X_val)
                accuracy_val.append(np.sum(y_val==Y_val)/float(len(Y_val)))
            print "Accuracy after iteration %i on training set: %f" %(i, accuracy[-1])
            if (i+1) % (print_interval*10)==0:
                print "The loss and accuracy in time so far on the training set are:"
                print loss
                print accuracy
                if training==True:
                    print "The loss and accuracy in time so far on the validation set are:"
                    print loss_val
                    print accuracy_val
        if i==num_passes-1:
            print "The final accuracy on the test set is "+str(accuracy[-1])
            if training==True:
                print "The final accuracy on the train set is "+str(accuracy_val[-1])

    return model

import pickle
# Load the input data in the training set
X = np.loadtxt(filename_x, delimiter=",") # load from text
#X = pickle.load(open('x.pkl',"rb"))
X=X.reshape(-1,4096)
# Filter the input values and convert them into binary ones
X[X<200]=0
X[X>200]=1
# Load the output results in the training set
y=[]
# Convert the output results to their corresponding class indices
with open(filename_y,'r') as fin:
    for x in fin:
        x=int(x)
        if x<19:
            temp=x
        elif x==20:
            temp=19
        elif x==21:
            temp=20
        elif x==24:
            temp=21
        elif x==25:
            temp=22
        elif x==27:
            temp=23
        elif x==28:
            temp=24
        elif x==30:
            temp=25
        elif x==32:
            temp=26
        elif x==35:
            temp=27
        elif x==36:
            temp=28
        elif x==40:
            temp=29
        elif x==42:
            temp=30
        elif x==45:
            temp=31
        elif x==48:
            temp=32
        elif x==49:
            temp=33
        elif x==54:
            temp=34
        elif x==56:
            temp=35
        elif x==63:
            temp=36
        elif x==64:
            temp=37
        elif x==72:
            temp=38
        elif x==81:
            temp=39
        y.append(temp)
y=np.array(y)
y.reshape(-1,1)
Y=y

# Build a model with a 4-dimensional hidden layer
num=0
time=0
total_size=len(X)
for hdim0 in hdim0_range: # The number of nodes
    for epsilon in epsilon_range: # learning rate for gradient descent
        #for time in range(5):
        # train_x,train
        num+=1
        test_x=X[int(len(X)*test_size*time):int(len(X)*test_size*(time+1))]
        train_x=X[int(len(X)*test_size*(time+1)):total_size]
        test_y=Y[int(len(X)*test_size*time):int(len(X)*test_size*(time+1))]
        train_y=Y[int(len(X)*test_size*(time+1)):total_size]
        print "Iteration "+str(num)+" starts----------------"
        print "Learning rate is "+str(epsilon)
        print "There are "+str(hdim0)+" neurons in the hidden layer"
        model = build_model(hdim0,hdim1, train_x, train_y, training=True, X_val=test_x, Y_val=test_y, print_loss=True,num_passes=num_passes_current,print_interval=print_interval_current)








