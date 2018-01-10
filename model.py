import tensorflow as tf
from lazy_property import lazy_property

'''
using structure from https://danijar.com/structuring-your-tensorflow-models/
'''
class Model:

    def __init__(self, feature, label):
        self.feature = feature
        self.label = label
        self.prediction
        self.optimize
        self.error

    @lazy_property
    def prediction(self):
        feature_size = 6
        label_size = 4
        layer_1_size = 10
        layer_2_size = 8

        with tf.variable_scope('layer_1') as scope:
            layer_1_weights = tf.Variable(tf.random_uniform(shape=[feature_size, layer_1_size], minval = 0.001, maxval = 0.01), name="weights")
            #layer_1_weights = tf.Variable(tf.constant(0.0, shape=[feature_size, layer_1_size]), name="weights")
            layer_1_biases = tf.Variable(tf.constant(0.001, shape = [layer_1_size]), name = "biases")
            layer_1 = tf.nn.relu(tf.matmul(self.feature, layer_1_weights) + layer_1_biases)
            
        with tf.variable_scope('layer_2') as scope:
            layer_2_weights = tf.Variable(tf.random_uniform(shape=[layer_1_size, layer_2_size], minval = 0.001, maxval = 0.01), name="weights")
            #layer_1_weights = tf.Variable(tf.constant(0.0, shape=[feature_size, layer_1_size]), name="weights")
            layer_2_biases = tf.Variable(tf.constant(0.001, shape = [layer_2_size]), name = "biases")
            layer_2 = tf.nn.relu(tf.matmul(layer_1, layer_2_weights) + layer_2_biases)

        with tf.variable_scope('layer_output') as scope:
            output_weights = tf.Variable(tf.random_uniform(shape = [layer_2_size, label_size], minval = 0.001, maxval = 0.01), name = "weights")
            output_biases = tf.Variable(tf.constant(0.001, shape = [label_size]), name = "biases")
            output = tf.matmul(layer_2, output_weights) + output_biases
            

        return output

    @lazy_property
    def optimize(self):
        learning_rate = 0.05
        cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.prediction, labels=self.label))
        #cost = tf.losses.mean_squared_error(self.prediction, self.label)
        #return tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost), cost
        return tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost), cost

    @lazy_property
    def error(self):
        correct_prediction = tf.equal(tf.argmax(self.prediction, 1), tf.argmax(self.label, 1))
        return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

'''
Layer_1 Activations (number of epochs to learn 2 different tactics (50 epoch increments))
    Sigmoid: 600 epochs (biases = 0.0)
    Sigmoid: 650 epochs (biases = 0.01)
    Tanh: 400 epochs (biases = 0.0)
    Elu: 300 - 400 epcohs (biases = 0.0)
    Relu: 300 - 400 epochs (biases = 0.0)
    Relu6 300 - 400 epochs (biases = 0.0)

Shuffling the data helps learning:
    Relu 200 - 250 epochs (biases = 0.0)

If the weights start at initialised at 0 then the training time doubles
'''
