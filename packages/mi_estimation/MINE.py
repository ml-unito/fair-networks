import tensorflow as tf
import tensorflow.contrib.layers as layers
import math
import numpy as np

class MINE():

    def __init__(self, x, y, hidden_neurons=50, learning_rate=1e-3):
        self.x = x 
        self.y = y
        self.hidden_neurons = hidden_neurons
        self.learning_rate = learning_rate
        self.create_mine()

    def create_mine(self):
        x_size = self.x.shape[1]
        y_size = self.y.shape[1]

        self.x_in = tf.placeholder(tf.float32, [None, x_size], name='mine_x_ph')
        self.y_in = tf.placeholder(tf.float32, [None, y_size], name='mine_y_ph')

        # shuffle and concatenate
        y_shuffle = tf.random_shuffle(self.y_in)
        x_conc = tf.concat([self.x_in, self.x_in], axis=0)
        y_conc = tf.concat([self.y_in, y_shuffle], axis=0)
        
        # propagate the forward pass
        layerx = layers.linear(x_conc, self.hidden_neurons)
        layery = layers.linear(y_conc, self.hidden_neurons)
        layer2 = tf.nn.relu(layerx + layery)
        self.output = layers.linear(layer2, 1)
        
        # split in T_xy and T_x_y predictions
        N_samples = tf.shape(self.x_in)[0]
        T_xy = self.output[:N_samples]
        T_x_y = self.output[N_samples:]
        
        # compute the negative loss (maximise loss == minimise -loss)
        self.neg_loss = -(tf.reduce_mean(T_xy, axis=0) - tf.math.log(tf.reduce_mean(tf.math.exp(T_x_y))))
        self.opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.neg_loss)


    def train(self, epochs=500, batch_size=64):
        num_batches = math.ceil(self.x.shape[0]/batch_size)
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        for epoch in range(epochs):
            x_epoch, y_epoch = self.shuffle_together(self.x, self.y)
            for i in range(num_batches):
                start = i * batch_size
                end   = (i+1) * batch_size
                x_batch = x_epoch[start:end]
                y_batch = y_epoch[start:end]
                feed_dict = {self.x_in:x_batch, self.y_in:y_batch}
                _, neg_l = self.sess.run([self.opt, self.neg_loss], feed_dict=feed_dict)

    def estimate(self, x, y):
        neg_l = self.sess.run(self.neg_loss, feed_dict={self.x_in: x, self.y_in: y})[0]
        return -neg_l

    def shuffle_together(self, a, b):
        assert len(a) == len(b)
        p = np.random.permutation(len(a))
        return a[p], b[p]

    def close_session(self):
        self.sess.close()