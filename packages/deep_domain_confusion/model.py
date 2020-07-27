import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
from fair.utils.loss_utils import mmd_loss
from fair.fn.model import Model

class DDCModel(Model):

    def __init__(self, options, optimizer):
        self._build(options, optimizer)
        self.s_variables = []  # placeholder so that the code does not complain

    def _build(self, options, optimizer):
        num_features = options.num_features
        num_s_labels = options.dataset.num_s_columns()

        num_y_labels = options.dataset.num_y_columns()
        is_conv_model = options.dataset == 'amazon' or options.dataset == 'other'

        self.noise_type = options.noise_type

        self.fairness_importance = tf.Variable(
            options.fairness_importance, name="fairness_importance", trainable=False)

        self.epoch = tf.get_variable(
            "epoch", shape=[1], initializer=tf.zeros_initializer)
        self.inc_epoch = self.epoch.assign(self.epoch + 1)

        self.x = tf.placeholder(tf.float32, shape=[None, num_features], name="x")
        self.noise = tf.placeholder(tf.float32, shape=[None, num_features], name="noise")
        self.y = tf.placeholder(tf.float32, shape=[None, num_y_labels], name="y")
        self.s = tf.placeholder(tf.float32, shape=[None, num_s_labels], name="s")
        s0_mask = tf.math.equal(self.s[:, 0], 0)
        s1_mask = tf.math.equal(self.s[:, 0], 1)

        h_layer, self.hidden_layers_variables = self._build_hidden_layers(self.x, options.hidden_layers)

        h_layer_source = tf.boolean_mask(h_layer, s0_mask)
        h_layer_target = tf.boolean_mask(h_layer, s1_mask)

        y_layer, self.class_layers_variables = self._build_layers(h_layer, "class", options.class_layers)

        self.model_last_hidden_layer = h_layer

        with tf.name_scope("y_out"):
            self.y_out = tf.layers.dense(y_layer, num_y_labels, activation=None,
                                         kernel_initializer=tf.truncated_normal_initializer(), name="y_out")

        with tf.name_scope("y_loss"):
            self.y_loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y, logits=self.y_out))
            self.y_train_loss_stat = tf.summary.scalar("y_train_softmax_loss", self.y_loss)
            self.y_val_loss_stat = tf.summary.scalar("y_val_softmax_loss", self.y_loss)

        with tf.name_scope("mmd_loss"):
            self.mmd_loss, loss_x, loss_y, loss_diff = mmd_loss(h_layer_source, h_layer_target, self.fairness_importance)
            self.mmd_train_loss_stat = tf.summary.scalar("mmd_loss_train", self.mmd_loss)
            self.mmd_val_loss_stat = tf.summary.scalar("mmd_loss_val", self.mmd_loss)
            self.mmd_x = tf.summary.scalar("mmd_loss_x", loss_x)
            self.mmd_y = tf.summary.scalar("mmd_loss_y", loss_y)
            self.mmd_diff = tf.summary.scalar("mmd_loss_diff", loss_diff)

        with tf.name_scope("y_accuracy"):
            y_correct_predictions = tf.cast(
                tf.equal(tf.argmax(self.y_out, 1), tf.argmax(self.y, 1)), "float")
            self.y_accuracy = tf.cast(
                tf.reduce_mean(y_correct_predictions), "float")
            self.y_train_accuracy_stat = tf.summary.scalar("y_train_accuracy", self.y_accuracy)
            self.y_val_accuracy_stat = tf.summary.scalar("y_val_accuracy", self.y_accuracy)

        self.train_step = optimizer.minimize(self.y_loss + self.mmd_loss)
        self.train_stats = tf.summary.merge([self.y_train_loss_stat, self.y_train_accuracy_stat,
                                             self.mmd_train_loss_stat, self.mmd_x, self.mmd_y, self.mmd_diff])
        self.val_stats = tf.summary.merge([self.y_val_loss_stat, self.y_val_accuracy_stat, self.mmd_val_loss_stat])

    def get_steps(self):
        return [self.train_step]

    def get_loggables(self):
        return {'y_loss': self.y_loss, 'mmd_loss': self.mmd_loss, 'y_acc': self.y_accuracy}
