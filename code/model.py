import tensorflow as tf
import time


class Model:
    def __init__(self, layers, optimizer, num_features, num_labels):
        self._build(layers, optimizer, num_features, num_labels)

    def _build(self, layers, optimizer, num_features, num_labels):
        self.x = tf.placeholder(tf.float32, shape=[None, num_features], name="x")
        self.y = tf.placeholder(tf.float32, shape=[None, num_labels], name="y")
        in_layer = self.x

        for index,layer in enumerate(layers):
            num_nodes, activation, initializer = layer
            with tf.name_scope("layer-%d" % (index+1)):
                in_layer = tf.layers.dense(in_layer, num_nodes, activation=activation, kernel_initializer = initializer())

        with tf.name_scope("out"):
            self.out = tf.layers.dense(in_layer, num_labels, activation=tf.nn.sigmoid, kernel_initializer = tf.truncated_normal_initializer())

        with tf.name_scope("loss"):
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y, logits=self.out))
            self.train_loss_stat = tf.summary.scalar("train_softmax_loss", self.loss)
            self.test_loss_stat = tf.summary.scalar("test_softmax_loss", self.loss)

        with tf.name_scope("accuracy"):
            correct_predictions = tf.cast(tf.equal(tf.argmax(self.out,1), tf.argmax(self.y,1)), "float")
            self.accuracy = tf.cast(tf.reduce_mean(correct_predictions), "float")
            self.train_accuracy_stat = tf.summary.scalar("train_accuracy", self.accuracy)
            self.test_accuracy_stat = tf.summary.scalar("test_accuracy", self.accuracy)


        with tf.name_scope("confusion_matrix"):
            predicted = tf.argmax(self.out, 1)
            actual = tf.argmax(self.y,1)
            TP = tf.count_nonzero(predicted * actual)
            TN = tf.count_nonzero((predicted - 1) * (actual - 1))
            FP = tf.count_nonzero(predicted * (actual - 1))
            FN = tf.count_nonzero((predicted - 1) * actual)
            self.confusion_matrix = (TP,TN,FP,FN)

        self.train_step = optimizer.minimize(self.loss)
        self.train_stats = tf.summary.merge([self.train_loss_stat, self.train_accuracy_stat])
        self.test_stats = tf.summary.merge([self.test_loss_stat, self.test_accuracy_stat])

        return self


    def eval_loss_and_accuracy(self, session, feed_dict):
        loss_val = session.run(self.loss, feed_dict=feed_dict)
        accuracy_val = session.run(self.accuracy, feed_dict=feed_dict)
        return (loss_val, accuracy_val)

    def print_loss_and_accuracy(self, session, train_feed_dict, test_feed_dict):
        loss_train_val, accuracy_train_val = self.eval_loss_and_accuracy(session, feed_dict = train_feed_dict )
        loss_test_val, accuracy_test_val = self.eval_loss_and_accuracy(session, feed_dict = test_feed_dict)

        print("")
        print("acc tr:%2.8f|acc te:%2.8f|loss tr:%2.8f|loss te:%2.8f" % (accuracy_train_val, accuracy_test_val, loss_train_val, loss_test_val))

    def print_confusion_matrix(self, session, feed_dict):
        (tp,tn,fp,fn) = session.run(self.confusion_matrix, feed_dict = feed_dict)

        print("|        |predicted +|predicted -|")
        print("|:------:|----------:|----------:|")
        print("|actual +|%11d|%11d|" % (tp,fn))
        print("|actual -|%11d|%11d|" % (fp,tn))
