from TensorModel import NeuralNetworkUtils
import tensorflow as tf

LR = 0.001


class convNet(NeuralNetworkUtils):
    def __init__(self, input_shape, labels_shape, parameters_shape):
        self.X, self.Y = self.create_placeholder(input_shape, labels_shape)
        self.parameters = self.create_parameters(parameters_shape)
        self.output = self.forward_propagation(self.X, self.parameters)
        self.cost = self.compute_cost(self.output, self.Y)
        self.train_step = self.optimizer('adam', LR)
        self.init = self.init_tensor_variables()

    def forward_propagation(self, X, parameters):
        """
        Initializes weight parameters to build a neural network with tensorflow.
        Arguments:
            parameters -- a dictionary of tensors containing the parameters Initialized

        Returns:
            output_Z --- last activation of the network
            """
        w1 = parameters['w1']
        w2 = parameters['w2']

        Z1 = tf.nn.conv2d(X, w1, strides=[1, 1, 1, 1], padding='SAME')
        A1 = tf.nn.relu(Z1)
        P1 = tf.nn.max_pool(A1, ksize=[1, 8, 8, 1], strides=[1, 8, 8, 1], padding='SAME')

        Z2 = tf.nn.conv2d(P1, w2, strides=[1, 1, 1, 1], padding='SAME')
        A2 = tf.nn.relu(Z2)
        P2 = tf.nn.max_pool(A2, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding='SAME')

        P2 = tf.contrib.layers.flatten(P2)
        Z3 = tf.contrib.layers.fully_connected(P2, 6, activation_fn=None)

        return Z3

    def compute_cost(self, logits, labels):
        """
        Compute the cost
            Arguments:
                logtis -- output of the network (output_Z)
                labels -- labels of the trainig (Y)

            Returns:
                output --- last activation of the network
        """
        cost = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))

        return cost


# conv1 = convNet([None, 64, 64, 3], [None, 6], {'w1': [4, 4, 3, 8], 'w2': [2, 2, 8, 16]})
# conv1.train(x_train, y_train, x_test, y_test, 1000)
