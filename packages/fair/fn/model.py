import tensorflow.compat.v1 as tf
import numpy as np
import time
import sys
from fair.utils.loss_utils import estimate_mean_and_variance


class Model:
    def __init__(self, options, optimizer):
        self._build(options, optimizer)

    def print_loss_and_accuracy(self, session, train_feed_dict, test_feed_dict):
        measures = [["y", [self.y_loss, self.y_accuracy]], ["s", [self.s_mean_loss, self.s_accuracy]]]

        print('|variable|acc. (train)|acc. (test)|loss (train)| loss(test)|')
        print('|:------:|-----------:|----------:|-----------:|----------:|')

        for name, loss_and_accuracy in measures:
            loss_train_val, accuracy_train_val = session.run(loss_and_accuracy, feed_dict = train_feed_dict)
            loss_test_val, accuracy_test_val = session.run(loss_and_accuracy, feed_dict = test_feed_dict)

            print("|%8s|     %2.5f|    %2.5f|     %2.5f|    %2.5f|" % (
                name,accuracy_train_val, accuracy_test_val, loss_train_val, loss_test_val))

    def print_confusion_matrix(self, session, feed_dict):
        (tp,tn,fp,fn) = session.run(self.confusion_matrix, feed_dict = feed_dict)

        print("|        |predicted +|predicted -|")
        print("|:------:|----------:|----------:|")
        print("|actual +|%11d|%11d|" % (tp,fn))
        print("|actual -|%11d|%11d|" % (fp,tn))

    def print_errors(self, session, feed_dict, out, exp_out):
        count = 0
        errors = 0
        predictions = session.run(out, feed_dict = feed_dict)
        for index in range(len(predictions)):
            count += 1
            pred = predictions[index,:]
            exp = feed_dict[exp_out][index,:]
            if np.argmax(pred) != np.argmax(exp):
                print("index: %d pred: %d expected: %d" % (index, np.argmax(pred), np.argmax(exp)))
                errors += 1
        print("accuracy: %f" % ((count - errors) / float(count)))


    def print_weight(self, session, index, variables=None):
        if variables == None:
            variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

        var = variables[index]
        var_value = session.run(var)
        print("var[{}]={}".format(var.name, var_value))

    def print_weights(self, session):
        variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        for index in range(len(variables)):
            self.print_weight(session, index, variables)

    

    # PRIVATE METHODS

    def _build(self, options, optimizer):
        num_features = options.num_features
        num_s_labels = options.dataset.num_s_columns()

        num_y_labels = options.dataset.num_y_columns()

        self.noise_type = options.noise_type

        self.fairness_importance = tf.Variable(
            options.fairness_importance, name="fairness_importance")

        self.epoch = tf.get_variable(
            "epoch", shape=[1], initializer=tf.zeros_initializer)
        self.inc_epoch = self.epoch.assign(self.epoch + 1)

        self.x     = tf.placeholder(tf.float32, shape=[None, num_features], name="x")
        self.noise = tf.placeholder(tf.float32, shape=[None, num_features], name="noise")
        self.y     = tf.placeholder(tf.float32, shape=[None, num_y_labels], name="y")
        self.s     = tf.placeholder(tf.float32, shape=[None, num_s_labels], name="s")

        self.h_random_mean, self.h_random_var = estimate_mean_and_variance(num_features)

        in_layer = self.x

        h_layer, self.hidden_layers_variables = self._build_hidden_layers(in_layer, options.hidden_layers)
        s_layer, self.sensible_layers_variables = self._build_layers(h_layer, "sensible", options.sensible_layers)
        y_layer, self.class_layers_variables = self._build_layers(h_layer, "class", options.class_layers)

        self.model_last_hidden_layer = h_layer

        with tf.name_scope("s_out"):
            self.s_out = tf.layers.dense(s_layer, num_s_labels, activation=None,
                                         kernel_initializer=tf.truncated_normal_initializer(), name="s_out")

        with tf.name_scope("y_out"):
            self.y_out = tf.layers.dense(y_layer, num_y_labels, activation=None,
                                         kernel_initializer=tf.truncated_normal_initializer(), name="y_out")

        with tf.name_scope("y_loss"):
            self.y_loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y, logits=self.y_out))
            self.y_train_loss_stat = tf.summary.scalar("y_train_softmax_loss", self.y_loss)
            self.y_val_loss_stat = tf.summary.scalar("y_val_softmax_loss", self.y_loss)

        with tf.name_scope("s_loss"):
            mean_kld, var_kld = tf.nn.moments(
                tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.s, logits=self.s_out), 0)
            self.s_var_loss = var_kld
            self.s_mean_loss = mean_kld
            self.s_train_loss_stat = tf.summary.scalar("s_train_softmax_loss", self.s_mean_loss)
            self.s_val_loss_stat = tf.summary.scalar("s_val_softmax_loss", self.s_mean_loss)

        with tf.name_scope("h_loss"):
            #self.h_loss = self.y_loss + self.fairness_importance * (tf.math.pow(self.s_mean_loss - self.h_random_mean, 2)
                                                                 #+  tf.math.pow(self.s_var_loss - self.h_random_var, 2))
            self.h_loss = self.y_loss - \
                (self.fairness_importance * self.s_mean_loss)
            self.h_train_loss_stat = tf.summary.scalar("h_train_loss", self.h_loss)
            self.h_val_loss_stat = tf.summary.scalar("h_val_loss", self.h_loss)

        with tf.name_scope("y_accuracy"):
            y_correct_predictions = tf.cast(
                tf.equal(tf.argmax(self.y_out, 1), tf.argmax(self.y, 1)), "float")
            self.y_accuracy = tf.cast(
                tf.reduce_mean(y_correct_predictions), "float")
            self.y_train_accuracy_stat = tf.summary.scalar("y_train_accuracy", self.y_accuracy)
            self.y_val_accuracy_stat = tf.summary.scalar("y_val_accuracy", self.y_accuracy)

        with tf.name_scope("s_accuracy"):
            s_correct_predictions = tf.cast(
                tf.equal(tf.argmax(self.s_out, 1), tf.argmax(self.s, 1)), "float")
            self.s_accuracy = tf.cast(tf.reduce_mean(s_correct_predictions), "float")
            self.s_train_accuracy_stat = tf.summary.scalar("s_train_accuracy", self.s_accuracy)
            self.s_val_accuracy_stat = tf.summary.scalar("s_val_accuracy", self.s_accuracy)

        with tf.name_scope("svc_accuracies"):
            self.s_svc_accuracy = tf.placeholder(tf.float32)
            self.y_svc_accuracy = tf.placeholder(tf.float32)
            self.s_svc_accuracy_stat = tf.summary.scalar("s_svc_accuracy", self.s_svc_accuracy)
            self.y_svc_accuracy_stat = tf.summary.scalar("y_svc_accuracy", self.y_svc_accuracy)

        with tf.name_scope("y_confusion_matrix"):
            predicted = tf.argmax(self.y_out, 1)
            actual = tf.argmax(self.y, 1)
            TP = tf.count_nonzero(predicted * actual)
            TN = tf.count_nonzero((predicted - 1) * (actual - 1))
            FP = tf.count_nonzero(predicted * (actual - 1))
            FN = tf.count_nonzero((predicted - 1) * actual)
            self.confusion_matrix = (TP, TN, FP, FN)

        self.y_variables = [self.class_layers_variables, tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, "y_out")]
        self.s_variables = [self.sensible_layers_variables, tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, "s_out")]

        self.h_train_step, self.y_train_step, self.s_train_step = self._create_train_steps_alt(optimizer)

        self.train_stats = tf.summary.merge([self.y_train_loss_stat, self.y_train_accuracy_stat, self.s_train_loss_stat,
                                             self.s_train_accuracy_stat, self.h_train_loss_stat])
        self.val_stats = tf.summary.merge([self.y_val_loss_stat, self.y_val_accuracy_stat, self.s_val_loss_stat,
                                            self.s_val_accuracy_stat, self.h_val_loss_stat])

        return self

    def _build_layers(self, in_layer, layer_name, layers):
        variables = []

        for index, layer in enumerate(layers):
            if layer[0] == 'i':
                pass
            else:
                in_layer, temp_variables = self._build_layer(
                    in_layer, layer_name, layer, index)
                variables.extend(temp_variables)

        return in_layer, variables

    def _build_hidden_layers(self, in_layer, hidden_layers):
        hidden_variables = []
        for i, layer in enumerate(hidden_layers):
            layer_type = layer[0]
            if layer_type == 'n':
                initializer = layer[3]
                in_layer, variables = self._build_layer_noise(in_layer, initializer, i+1)
            elif layer_type == 'w':
                initializer = layer[3]
                in_layer, variables = self._build_layer(in_layer, "hidden-whiteout", layer, i+1, whiteout=True)
            else:
                in_layer, variables = self._build_layer(in_layer, "hidden", layer, i+1)
            hidden_variables.extend(variables)
        return in_layer, hidden_variables

    def _build_layer(self, in_layer, layer_name, layer, index, whiteout=False):
        layer_variables = []

        _, num_nodes, activation, initializers = layer
        with tf.name_scope("%s-layer-%d" % (layer_name, index+1)):
            in_layer = tf.layers.dense(in_layer, 
                            num_nodes, activation=activation, 
                            kernel_initializer=initializers[0](), 
                            bias_initializer=initializers[1](), 
                            name='%s-layer-%d' % (layer_name, index+1)
                        )

            with tf.variable_scope("%s-layer-%d" % (layer_name, index+1), reuse=True):
                w = tf.get_variable("kernel")
                b = tf.get_variable("bias")
                layer_variables.extend([w, b])

            if whiteout:
                with tf.variable_scope("%s-layer-%d-noiseparams" % (layer_name, index+1), reuse=True):
                    gaussian_noise = tf.random.normal(tf.shape(w))
                    w_abs = tf.math.abs(w)
                    gamma = 1
                    _lambda = 1
                    whiteout_term = gaussian_noise * (tf.math.pow(w_abs, -gamma) + _lambda)
                    w = w + whiteout_term

        return in_layer, layer_variables

    def _build_layer_conv_2d(self, in_layer, parameters):
        layer_variables = []
        layer = tf.layers.conv2d(in_layer, parameters['n_filters'], parameters['kernel_size'],
                                 strides=parameters['strides'], kernel_initializer=parameters['kernel_initializer'](),
                                 name='{}-conv-{}'.format(parameters['name'], parameters['index']+1))
        
        with tf.variable_scope('{}-conv-{}'.format(parameters['name'], parameters['index']+1), reuse=True):
                w = tf.get_variable("kernel")
                b = tf.get_variable("bias")
                layer_variables.extend([w, b]) 
        return layer, layer_variables

    def _build_layer_noise(self, in_layer, initializer, index):
        variables = []

        with tf.variable_scope("%s-layer-%d" % ('noise', index)):
            alpha = tf.get_variable("alpha", dtype=tf.float32, 
                            shape=[in_layer.get_shape()[1]], 
                            initializer=initializer[0]())

            w_beta = tf.get_variable("w-beta", dtype=tf.float32, 
                            shape=[in_layer.get_shape()[1]], 
                            initializer=initializer[1]())

            beta = tf.multiply(self.noise, w_beta)
            
            if self.noise_type == 'default':
                out = tf.multiply(in_layer, alpha) + beta
            elif self.noise_type == 'sigmoid_full':
                out = tf.nn.sigmoid(tf.multiply(in_layer, alpha) + beta)
            elif self.noise_type == 'sigmoid_sep':
                out = tf.nn.sigmoid(tf.multiply(in_layer, alpha)) + tf.nn.sigmoid(beta)
            elif self.noise_type == 'sigmoid_sep_2':
                out = (tf.nn.sigmoid(tf.multiply(in_layer, alpha)) + tf.nn.sigmoid(beta)) / 2

            variables.extend([alpha, w_beta])
        return out, variables

    def _build_layer_whiteout(self, in_layer, initializer, index):
        layer_variables = []

        _, num_nodes, activation, initializers = layer
        with tf.name_scope("%s-layer-%d" % (layer_name, index+1)):
            in_layer = tf.layers.dense(in_layer, 
                            num_nodes, activation=activation, 
                            kernel_initializer=initializers[0](), 
                            bias_initializer=initializers[1](), 
                            name='%s-layer-%d' % (layer_name, index+1)
                        )

            with tf.variable_scope("%s-layer-%d" % (layer_name, index+1), reuse=True):
                w = tf.get_variable("kernel")
                b = tf.get_variable("bias")
                layer_variables.extend([w, b])

        return in_layer, layer_variables

    def _create_train_steps(self, optimizer):
        h_grads_vars_s = optimizer.compute_gradients(
            self.s_mean_loss, var_list=self.hidden_layers_variables)
        h_grads_vars_s = [(self.fairness_importance * -gv[0], gv[1])
                          for gv in h_grads_vars_s]
        h_grads_vars_y = optimizer.compute_gradients(
            self.y_loss, var_list=self.hidden_layers_variables)
        y_grads = optimizer.compute_gradients(
            self.y_loss, var_list=self.y_variables)
        s_grads = optimizer.compute_gradients(
            self.s_mean_loss, var_list=self.s_variables)
        h_s_step = optimizer.apply_gradients(h_grads_vars_s)
        h_y_step = optimizer.apply_gradients(h_grads_vars_y)
        self.h_grads = h_grads_vars_s + h_grads_vars_y
        h_train_step = tf.group(h_s_step, h_y_step)
        y_train_step = optimizer.apply_gradients(y_grads)
        s_train_step = optimizer.apply_gradients(s_grads)
        return h_train_step, y_train_step, s_train_step

    def _create_train_steps_alt(self, optimizer):
        """
        An alternative way of creating training steps which works in the following way:
        1. the gradients for y_vars and h_vars are computed normally and the parameters are updated
        2. the gradients for s_vars and h_vars are computed and reverted (for h_vars) and the parameters are updated
        This assumes that the steps are called in the following order: y_step, s_step. h_step is a no_op and
        does not matter.
        """
        y_grads = optimizer.compute_gradients(
            self.y_loss, var_list=self.y_variables.extend(self.hidden_layers_variables))
        s_grads = optimizer.compute_gradients(
            self.s_mean_loss, var_list=self.s_variables.extend(self.hidden_layers_variables))
        h_grads = optimizer.compute_gradients(self.s_mean_loss, var_list=self.hidden_layers_variables)
        h_grads = [(self.fairness_importance * -gv[0], gv[1]) for gv in h_grads]
        h_step_tmp = optimizer.apply_gradients(h_grads)
        y_train_step = optimizer.apply_gradients(y_grads)
        s_train_step = optimizer.apply_gradients(s_grads)
        s_train_step = tf.group(s_train_step, h_step_tmp)
        h_train_step = tf.no_op()
        return h_train_step, y_train_step, s_train_step
