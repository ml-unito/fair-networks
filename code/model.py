import tensorflow as tf
import time


class Model:
    def __init__(self, layers, optimizer, num_features, num_y_labels, num_s_labels):
        self._build(layers, optimizer, num_features, num_y_labels, num_s_labels)

    def _build(self, layers, optimizer, num_features, num_y_labels, num_s_labels):
        self.x = tf.placeholder(tf.float32, shape=[None, num_features], name="x")
        self.y = tf.placeholder(tf.float32, shape=[None, num_y_labels], name="y")
        self.s = tf.placeholder(tf.float32, shape=[None, num_s_labels], name="s")
        in_layer = self.x

        self.shared_layers_variables = []

        for index,layer in enumerate(layers):
            num_nodes, activation, initializer = layer
            with tf.name_scope("layer-%d" % (index+1)):
                in_layer = tf.layers.dense(in_layer, num_nodes, activation=activation, kernel_initializer = initializer(), name='layer-%d' % (index+1))
            self.shared_layers_variables.append(
                    tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "layer-%d" % (index+1))[0])

        with tf.name_scope("s_out"):
            self.s_out = tf.layers.dense(in_layer, num_s_labels, activation=tf.nn.sigmoid, kernel_initializer = tf.truncated_normal_initializer(), name="s_out")

        with tf.name_scope("y_out"):
            self.y_out = tf.layers.dense(in_layer, num_y_labels, activation=tf.nn.sigmoid, kernel_initializer = tf.truncated_normal_initializer(), name="y_out")

        with tf.name_scope("y_loss"):
            self.y_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y, logits=self.y_out))
            self.y_train_loss_stat = tf.summary.scalar("y_train_softmax_loss", self.y_loss)
            self.y_test_loss_stat = tf.summary.scalar("y_test_softmax_loss", self.y_loss)

        with tf.name_scope("s_loss"):
            self.s_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.s, logits=self.s_out))
            self.s_train_loss_stat = tf.summary.scalar("s_train_softmax_loss", self.s_loss)
            self.s_test_loss_stat = tf.summary.scalar("s_test_softmax_loss", self.s_loss)

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

        with tf.name_scope("y_confusion_matrix"):
            predicted = tf.argmax(self.y_out, 1)
            actual = tf.argmax(self.y,1)
            TP = tf.count_nonzero(predicted * actual)
            TN = tf.count_nonzero((predicted - 1) * (actual - 1))
            FP = tf.count_nonzero(predicted * (actual - 1))
            FN = tf.count_nonzero((predicted - 1) * actual)
            self.confusion_matrix = (TP,TN,FP,FN)

        s_branch_variables = [vars for vars in self.shared_layers_variables]
        y_branch_variables = s_branch_variables.copy()
        y_branch_variables.extend(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "y_out"))
        s_branch_variables.extend(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "s_out"))

        self.y_train_step = optimizer.minimize(self.y_loss, var_list=[y_branch_variables])
        self.s_train_step = optimizer.minimize(self.s_loss, var_list=[s_branch_variables])

        self.train_stats = tf.summary.merge([self.y_train_loss_stat, self.y_train_accuracy_stat, self.s_train_loss_stat, self.s_train_accuracy_stat])
        self.test_stats = tf.summary.merge([self.y_test_loss_stat, self.y_test_accuracy_stat, self.s_test_loss_stat, self.s_test_accuracy_stat])
        
        return self


    def print_loss_and_accuracy(self, session, train_feed_dict, test_feed_dict):
        measures = [["y", [self.y_loss, self.y_accuracy]], ["s", [self.s_loss, self.s_accuracy]]]

        for name, loss_and_accuracy in measures:
            loss_train_val, accuracy_train_val = session.run(loss_and_accuracy, feed_dict = train_feed_dict)
            loss_test_val, accuracy_test_val = session.run(loss_and_accuracy, feed_dict = test_feed_dict)

            print("%s:" % (name))
            print("acc tr:%2.8f|acc te:%2.8f|loss tr:%2.8f|loss te:%2.8f" % (accuracy_train_val, accuracy_test_val, loss_train_val, loss_test_val))

    def print_confusion_matrix(self, session, feed_dict):
        (tp,tn,fp,fn) = session.run(self.confusion_matrix, feed_dict = feed_dict)

        print("|        |predicted +|predicted -|")
        print("|:------:|----------:|----------:|")
        print("|actual +|%11d|%11d|" % (tp,fn))
        print("|actual -|%11d|%11d|" % (fp,tn))
