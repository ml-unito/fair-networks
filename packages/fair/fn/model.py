import tensorflow as tf
import numpy as np
import time
import sys


class Model:
    def __init__(self, options, optimizer):
        self._build(options, optimizer)

    def build_layer(self, in_layer, layer_name, layers):
        layer_variables = []

        for index,layer in enumerate(layers):
            num_nodes, activation, initializer = layer
            with tf.name_scope("%s-layer-%d" % (layer_name, index+1)):
                in_layer = tf.layers.dense(in_layer, num_nodes, activation=activation, kernel_initializer = initializer(), name='%s-layer-%d' % (layer_name, index+1))
                with tf.variable_scope("%s-layer-%d" % (layer_name, index+1), reuse=True):
                    w = tf.get_variable("kernel")
                    b = tf.get_variable("bias")
                    layer_variables.extend([w, b])

        return in_layer, layer_variables

    def build_layer_random(self, in_layer, hidden_layers, random_units):
        try:
            assert len(random_units) == len(hidden_layers)
        except AssertionError:
            print('len(random_units) != len(hidden_layers): {} != {}'.format(len(random_units), len(hidden_layers)))
            sys.exit(1)

        hidden_layers_variables = []
        
        for index, hidden_layer in enumerate(hidden_layers):
            num_nodes_in = in_layer.shape[1]
            num_nodes_out, activation, initializer = hidden_layer
            with tf.variable_scope("%s-layer-%d" % ("hidden", index+1)):
                w = tf.get_variable(name="kernel", initializer=initializer(), shape=[num_nodes_in+random_units[index], num_nodes_out])
                b = tf.get_variable(name="bias", initializer=initializer(), shape=[num_nodes_out])
                x_rand = tf.random_normal(name="{}-layer-{}-random".format("hidden", index+1), shape=[tf.shape(in_layer)[0], random_units[index]])
                in_layer = tf.concat([in_layer, x_rand], axis=1)
                layer_out = activation(tf.matmul(in_layer, w) + b)
            in_layer = layer_out
            hidden_layers_variables.extend([w, b])
        
        return in_layer, hidden_layers_variables

    def _build(self, options, optimizer):
        num_features = options.num_features
        num_s_labels = options.dataset.num_s_columns()
        num_y_labels = options.dataset.num_y_columns()

        self.fairness_importance = tf.Variable(options.fairness_importance, name="fairness_importance")

        self.epoch = tf.get_variable("epoch", shape=[1], initializer=tf.zeros_initializer)
        self.inc_epoch = self.epoch.assign(self.epoch + 1)

        self.x = tf.placeholder(tf.float32, shape=[None, num_features], name="x")
        self.y = tf.placeholder(tf.float32, shape=[None, num_y_labels], name="y")
        self.s = tf.placeholder(tf.float32, shape=[None, num_s_labels], name="s")

        in_layer = self.x 

        if options.random_units == [0]:
            h_layer, self.hidden_layers_variables = self.build_layer(in_layer, "hidden", options.hidden_layers)
        else:
            h_layer, self.hidden_layers_variables  = self.build_layer_random(in_layer, options.hidden_layers, options.random_units)

        s_layer, self.sensible_layers_variables = self.build_layer(h_layer, "sensible", options.sensible_layers)
        y_layer, self.class_layers_variables    = self.build_layer(h_layer, "class", options.class_layers)
        

        self.model_last_hidden_layer = h_layer
        # self.model_last_hidden_layer = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "hidden-layer-%d" % (len(options.hidden_layers)))[0]

        with tf.name_scope("s_out"):
            self.s_out = tf.layers.dense(s_layer, num_s_labels, activation=None, kernel_initializer = tf.truncated_normal_initializer(), name="s_out")

        with tf.name_scope("y_out"):
            self.y_out = tf.layers.dense(y_layer, num_y_labels, activation=None, kernel_initializer = tf.truncated_normal_initializer(), name="y_out")

        with tf.name_scope("y_loss"):
            self.y_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y, logits=self.y_out))
            self.y_train_loss_stat = tf.summary.scalar("y_train_softmax_loss", self.y_loss)
            self.y_test_loss_stat = tf.summary.scalar("y_test_softmax_loss", self.y_loss)

        with tf.name_scope("s_loss"):
            mean_kld, var_kld = tf.nn.moments(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.s, logits=self.s_out), 0)
            self.s_var_loss = var_kld
            self.s_mean_loss = mean_kld
            self.s_train_loss_stat = tf.summary.scalar("s_train_softmax_loss", self.s_mean_loss)
            self.s_test_loss_stat = tf.summary.scalar("s_test_softmax_loss", self.s_mean_loss)

        with tf.name_scope("h_loss"):
            if options.var_loss:
                self.h_loss = self.y_loss - self.fairness_importance * self.s_var_loss
            else:
                self.h_loss = self.y_loss - self.fairness_importance * self.s_mean_loss
            self.h_train_loss_stat = tf.summary.scalar("h_train_loss", self.h_loss)
            self.h_test_loss_stat = tf.summary.scalar("h_test_loss", self.h_loss)

        with tf.name_scope("y_accuracy"):
            y_correct_predictions = tf.cast(tf.equal(tf.argmax(self.y_out,1), tf.argmax(self.y,1)), "float")
            self.y_accuracy = tf.cast(tf.reduce_mean(y_correct_predictions), "float")
            self.y_train_accuracy_stat = tf.summary.scalar("y_train_accuracy", self.y_accuracy)
            self.y_test_accuracy_stat = tf.summary.scalar("y_test_accuracy", self.y_accuracy)

        with tf.name_scope("s_accuracy"):
            s_correct_predictions = tf.cast(tf.equal(tf.argmax(self.s_out,1), tf.argmax(self.s,1)), "float")
            self.s_accuracy = tf.cast(tf.reduce_mean(s_correct_predictions), "float")
            self.s_train_accuracy_stat = tf.summary.scalar("s_train_accuracy", self.s_accuracy)
            self.s_test_accuracy_stat = tf.summary.scalar("s_test_accuracy", self.s_accuracy)

        with tf.name_scope("svc_accuracies"):
            self.s_svc_accuracy = tf.placeholder(tf.float32)
            self.y_svc_accuracy = tf.placeholder(tf.float32)
            self.s_svc_accuracy_stat = tf.summary.scalar("s_svc_accuracy", self.s_svc_accuracy)
            self.y_svc_accuracy_stat = tf.summary.scalar("y_svc_accuracy", self.y_svc_accuracy)

        with tf.name_scope("y_confusion_matrix"):
            predicted = tf.argmax(self.y_out, 1)
            actual = tf.argmax(self.y,1)
            TP = tf.count_nonzero(predicted * actual)
            TN = tf.count_nonzero((predicted - 1) * (actual - 1))
            FP = tf.count_nonzero(predicted * (actual - 1))
            FN = tf.count_nonzero((predicted - 1) * actual)
            self.confusion_matrix = (TP,TN,FP,FN)

        self.y_variables = [self.hidden_layers_variables, self.class_layers_variables, tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "y_out")]
        self.s_variables = [self.sensible_layers_variables, tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "s_out")]

        self.h_grads = optimizer.compute_gradients(self.h_loss, self.hidden_layers_variables)
        self.h_train_step = optimizer.apply_gradients(self.h_grads)
        self.s_grads = optimizer.compute_gradients(self.s_mean_loss, self.sensible_layers_variables)
        self.s_train_step = optimizer.apply_gradients(self.s_grads)
        self.y_grads = optimizer.compute_gradients(self.y_loss, self.class_layers_variables)
        self.y_train_step = optimizer.apply_gradients(self.y_grads)


        self.h_grads_stats = tf.summary.merge([tf.summary.histogram("%s-h-grad" % v.name, g) for g, v in self.h_grads])
        self.h_weight_stats = tf.summary.merge([tf.summary.histogram("%s-h-weights" % v.name, v) for g, v in self.h_grads])
        self.s_grads_stats = tf.summary.merge([tf.summary.histogram("%s-s-grad" % v.name, g) for g, v in self.s_grads])
        self.s_weight_stats = tf.summary.merge([tf.summary.histogram("%s-s-weights" % v.name, v) for g, v in self.s_grads])
        self.y_grads_stats = tf.summary.merge([tf.summary.histogram("%s-y-grad" % v.name, g) for g, v in self.y_grads])
        self.y_weight_stats = tf.summary.merge([tf.summary.histogram("%s-y-weights" % v.name, v) for g, v in self.y_grads])

        self.train_stats = tf.summary.merge([self.y_train_loss_stat, self.y_train_accuracy_stat, self.s_train_loss_stat, 
                            self.s_train_accuracy_stat, self.h_train_loss_stat])
        self.test_stats = tf.summary.merge([self.y_test_loss_stat, self.y_test_accuracy_stat, self.s_test_loss_stat, 
                            self.s_test_accuracy_stat, self.h_test_loss_stat])
        self.grad_stats = tf.summary.merge([self.h_grads_stats, self.s_grads_stats, self.y_grads_stats])
        self.weight_stats = tf.summary.merge([self.h_weight_stats, self.s_weight_stats, self.y_weight_stats])

        return self


    def print_loss_and_accuracy(self, session, train_feed_dict, test_feed_dict):
        measures = [["y", [self.y_loss, self.y_accuracy]], ["s", [self.s_mean_loss, self.s_accuracy]]]

        print('|variable|acc. (train)|acc. (test)|loss (train)| loss(test)|')
        print('|:------:|-----------:|----------:|-----------:|----------:|')

        for name, loss_and_accuracy in measures:
            loss_train_val, accuracy_train_val = session.run(loss_and_accuracy, feed_dict = train_feed_dict)
            loss_test_val, accuracy_test_val = session.run(loss_and_accuracy, feed_dict = test_feed_dict)

            print("|%8s|     %2.5f|    %2.5f|     %2.5f|    %2.5f|" % (name,accuracy_train_val, accuracy_test_val, loss_train_val, loss_test_val))

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
