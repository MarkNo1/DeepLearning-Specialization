from abc import ABC, abstractmethod
from tensorflow.python.framework import ops
import tensorflow as tf


class NeuralNetworkUtils(ABC):
    def __init__(self):
        pass

    def create_placeholder(self, input, label):
        """
        Creates the placeholders for the tensorflow session.

        Arguments:
        input -- Input shape dimension Ex: [None, image_H, image_W, image_C]
        labes -- Label dimension Ex: [None, labels_dimension]

        Returns:
        X -- placeholder for the data input, of shape [None, n_H0, n_W0, n_C0] and dtype "float"
        Y -- placeholder for the input labels, of shape [None, n_y] and dtype "float"
        """
        X = tf.placeholder(tf.float32, input, 'X')
        Y = tf.placeholder(tf.float32, label, 'Y')
        return X, Y

    def create_parameters(self, parameters_shape):
        """
        Initializes weight parameters to build a neural network with tensorflow.
        Arguments:
            parameters_shape -- dictionary of parameters shape. Ex : {'W1':[4,4,3,8], ... }

        Returns:
            parameters -- a dictionary of tensors containing the parameters Initialized
        """
        parameters = dict()
        for name, shape in parameters_shape.items():
            param = tf.get_variable(
                name, shape, initializer=tf.contrib.layers.xavier_initializer(seed=0))
            parameters[name] = param

        return parameters

    @abstractmethod
    def forward_propagation(self, X, parameters):
        """
        Initializes weight parameters to build a neural network with tensorflow.
        Arguments:
            parameters -- a dictionary of tensors containing the parameters Initialized

        Returns:
            output_Z --- last activation of the network
            """
        return NotImplemented

    @abstractmethod
    def compute_cost(self, logits, labels):
        """
        Compute the cost
            Arguments:
                logtis -- output of the network (output_Z)
                labels -- labels of the trainig (Y)

            Returns:
                output --- last activation of the network
        """
        return NotImplemented

    def optimizer(self, name, learning_rate):
        if 'adam' in name:
            return tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost)

    def prediction(self, output_Z, labels):
        """
        Calculate the correct prediction
            Arguments:
                output_Z -- output of the network
                labels -- labels of the trainig (Y)

        Returns:

                correct_prediction --- tensor op for correct prediction
                accuracy -- tensor op for accuracy
        """
        predict_op = tf.argmax(output_Z, 1)
        correct_prediction = tf.equal(predict_op, tf.argmax(labels, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        return correct_prediction, accuracy

    def init_tensor_variables(self):
        init = tf.global_variables_initializer()
        return init

    def train(self, x_train, y_train, x_test, y_test, epoches, reset=False, init=True):
        if reset:
            ops.reset_default_graph()
        if init:
            with tf.Session() as sess:
                sess.run(self.init)

                for epoch in range(epoches):
                    minibatches = random_mini_batches(
                        X_train, Y_train, minibatch_size, seed)

                    for minibatch in minibatches:
                        (minibatch_X, minibatch_Y) = minibatch
                        _, temp_cost = sess.run([optimizer, cost], feed_dict={
                                                X: minibatch_X, Y: minibatch_Y})

                        minibatch_cost += temp_cost / num_minibatches

                    if print_cost == True and epoch % 5 == 0:
                        print("Cost after epoch %i: %f" % (epoch, minibatch_cost))
                    if print_cost == True:
                        costs.append(minibatch_cost)
        return cost
