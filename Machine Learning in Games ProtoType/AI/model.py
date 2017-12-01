#Model.py
import tensorflow as tf
import numpy as np



#X = tf.placeholder(tf.float32, [None, 3])
#Y = tf.placeholder(tf.float32, [None, 4])#[action1, action2, action3, action4][1,0,0,0]

def simple_model(X):
    layer1 = tf.contrib.layers.fully_connected(
        inputs = X,
        num_outputs = 6,
        activation_fn = tf.nn.sigmoid,
        weights_initializer = tf.random_uniform_initializer(minval=0.001, maxval= 0.01), #constant_initializer(0.01), #tf.contrib.layers.variance_scaling_initializer(factor=400, uniform=True),#tf.contrib.layers.xavier_initializer(),
        #biases_initializer = tf.constant_initializer(0.01) #tf.zeros_initializer()
        )

    #output = tf.contrib.layers.fully_connected(
    #    inputs = layer1,
    #    num_outputs = 4,
    #    activation_fn = tf.nn.sigmoid,
    #    weights_initializer = tf.constant_initializer(0.01),
    #    biases_initializer = tf.constant_initializer(0.01)
    #    )

    output = tf.contrib.layers.fully_connected(
        inputs = layer1,
        num_outputs = 4,
        activation_fn = None,
        weights_initializer = tf.random_uniform_initializer(minval=0.001, maxval= 0.01), 
        )

    return output