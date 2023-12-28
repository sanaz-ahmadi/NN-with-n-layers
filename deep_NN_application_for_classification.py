#!/usr/bin/env python
# coding: utf-8

# # Building cat/not-a-cat classifier (Deep Neural Network for Image Classification)
# 
# - Build and train a deep L-layer neural network, and apply it to supervised learning
# 
# **first step: Building a deep neural network for image classification.**
# 
# - Using non-linear units like ReLU to improve our model
# - Building a deeper neural network (with more than 1 hidden layer)
# - Implementing an easy-to-use neural network class
# 
# *Here's an outline of the steps (first step):*
# - Initialize the parameters an $L$-layer neural network
# - Implement the forward propagation module 
# - The ACTIVATION function is: relu/sigmoid
# - Stack the [LINEAR->RELU] forward function L-1 time (for layers 1 through L-1) and add a [LINEAR->SIGMOID] at the end (for the final layer $L$, resulting in $Z^{[l]}$)). This gives you a new L_model_forward function.
# - Compute the loss
# - Implement the backward propagation module
# - The gradient of the ACTIVATION function is: relu_backward/sigmoid_backward 
# - Stack [LINEAR->RELU] backward L-1 times and add [LINEAR->SIGMOID] backward in a new L_model_backward function
# - Finally, update the parameters

# In[18]:


import time
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage
from dnn_app_utils_v3 import *
from dnn_app_utils_v3 import load_data
from public_tests import *

get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

np.random.seed(1)


# ## 1- Load and Process the Dataset
# 
# - a training set of `m_train` images labelled as cat (1) or non-cat (0)
# - a test set of `m_test` images labelled as cat and non-cat
# - each image is of shape (num_px, num_px, 3) where 3 is for the 3 channels (RGB).

# In[19]:


train_x_orig, train_y, test_x_orig, test_y, classes = load_data()


# In[20]:


# Explore our dataset 
m_train = train_x_orig.shape[0]
num_px = train_x_orig.shape[1]
m_test = test_x_orig.shape[0]

print ("Number of training examples: " + str(m_train))
print ("Number of testing examples: " + str(m_test))
print ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
print ("train_x_orig shape: " + str(train_x_orig.shape))
print ("train_y shape: " + str(train_y.shape))
print ("test_x_orig shape: " + str(test_x_orig.shape))
print ("test_y shape: " + str(test_y.shape))


# In[21]:


# Reshape the training and test examples 
#combining three channels into a vector (red, blue, green)
#standardize the images before feeding them to the network
train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T   # The "-1" makes reshape flatten the remaining dimensions
test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

# Standardize data to have feature values between 0 and 1.
train_x = train_x_flatten/255.
test_x = test_x_flatten/255.

print ("train_x's shape: " + str(train_x.shape))
print ("test_x's shape: " + str(test_x.shape))


# ## 2- Model Architecture
# ### L-layer Deep Neural Network:
# - The model can be summarized as: [LINEAR -> RELU] $\times$ (L-1) -> LINEAR -> SIGMOID
# 1. Initializing parameters / Define hyperparameters
# 2. Loop for num_iterations:
#     a. Forward propagation
#     b. Compute cost function
#     c. Backward propagation
#     d. Update parameters (using parameters, and grads from backprop) 
# 3. Using trained parameters to predict labels

# In[22]:


### CONSTANTS ###
layers_dims = [12288, 20, 7, 5, 1] #  4-layer model


# In[23]:


# L_layer_model

def L_layer_model(X, Y, layers_dims, learning_rate = 0.0075, num_iterations = 3000, print_cost=False):
    """
    Implements a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.
    
    Arguments:
    X -- input data, of shape (n_x, number of examples)
    Y -- true "label" vector (containing 1 if cat, 0 if non-cat), of shape (1, number of examples)
    layers_dims -- list containing the input size and each layer size, of length (number of layers + 1).
    learning_rate -- learning rate of the gradient descent update rule
    num_iterations -- number of iterations of the optimization loop
    print_cost -- if True, it prints the cost every 100 steps
    
    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """

    np.random.seed(1)
    costs = []                         # keep track of cost
    
    # Parameters initialization.
    parameters = initialize_parameters_deep(layers_dims)
    
    # Loop (gradient descent)
    for i in range(0, num_iterations):

        # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
        AL, caches = L_model_forward(X, parameters)
        cost = compute_cost(AL, Y)
    
        # Backward propagation
        grads = L_model_backward(AL, Y, caches)
        
        # Update parameters.
        parameters = update_parameters(parameters, grads, learning_rate)
        
        # Print the cost every 100 iterations
        if print_cost and i % 100 == 0 or i == num_iterations - 1:
            print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))
        if i % 100 == 0 or i == num_iterations:
            costs.append(cost)
    
    return parameters, costs


# In[24]:


parameters, costs = L_layer_model(train_x, train_y, layers_dims, num_iterations = 1, print_cost = False)

print("Cost after first iteration: " + str(costs[0]))

L_layer_model_test(L_layer_model)


# ## 3 - Train the model 
# 
# training your model as a 4-layer neural network, and the cost should decrease on every iteration.

# In[25]:


parameters, costs = L_layer_model(train_x, train_y, layers_dims, num_iterations = 2500, print_cost = True)


# In[26]:


pred_train = predict(train_x, train_y, parameters)


# In[27]:


pred_test = predict(test_x, test_y, parameters)


# ##  4 - Results Analysis
# The following code will show a few mislabeled images. 

# In[28]:


print_mislabeled_images(classes, test_x, test_y, pred_test)


# **A few types of images the model tends to do poorly on include:** 
# - Cat body in an unusual position
# - Cat appears against a background of a similar color
# - Unusual cat color and species
# - Camera Angle
# - Brightness of the picture
# - Scale variation (cat is very large or small in image) 
