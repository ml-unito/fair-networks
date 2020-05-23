#!/usr/bin/python

import sys
import numpy as np
import tensorflow.compat.v1 as tf
import time
import pandas
import sklearn.model_selection as ms

EPOCHS = 100
BATCH_SIZE = 100
RANDOM_SEED = 42


class Model:
    def __init__(self, num_features, num_s):
        self.x = tf.placeholder(tf.float32, shape=[None, num_features])
        self.s = tf.placeholder(tf.float32, shape=[None, num_s])

        self.in_layer = self.x
        self.h_layer = tf.layers.dense(self.in_layer, 10, kernel_initializer=tf.truncated_normal_initializer())
        self.out = tf.layers.dense(self.h_layer, num_s, kernel_initializer=tf.truncated_normal_initializer())

        self.loss = tf.losses.sigmoid_cross_entropy(self.s, self.out)

        equal_check = tf.equal(tf.argmax(self.s, axis=1), tf.argmax(self.out, axis=1))
        self.accuracy = tf.reduce_mean( tf.cast(equal_check, tf.float32) )




file = sys.argv[1]

df = pandas.read_csv(file)
y_columns = df.filter(regex=("y.*")).values
s_columns = df.filter(regex=("s.*")).values
h_columns = df.filter(regex=("h.*")).values

h_train, h_test, s_train, s_test, y_train, y_test = ms.train_test_split(
    h_columns, s_columns, y_columns, random_state=RANDOM_SEED)


model = Model(h_columns.shape[1], s_columns.shape[1])

optimizer = tf.train.AdagradOptimizer(1.0)
train_step = optimizer.minimize(model.loss)

init = tf.global_variables_initializer()
session = tf.Session()
session.run(init)

for epoch in range(EPOCHS):
    batch_pos = 0
    while batch_pos < len(h_train):
        batch_x = h_train[batch_pos:(batch_pos+BATCH_SIZE)]
        batch_s = s_train[batch_pos:(batch_pos+BATCH_SIZE)]
        batch_pos += BATCH_SIZE

        session.run(train_step, feed_dict={model.x: batch_x, model.s: batch_s })

    print("loss:" + str(session.run(model.loss, feed_dict={model.x: batch_x, model.s: batch_s })))
    print("train:" + str(session.run(model.accuracy, feed_dict={model.x: h_train, model.s: s_train })))
    print("test:" + str(session.run(model.accuracy, feed_dict={model.x: h_test, model.s: s_test })))



