import sys
import numpy as np
import tensorflow as tf
import time
import pandas
import sklearn.model_selection as ms

sys.path.append('code')

EPOCHS = 100
BATCH_SIZE = 100


class Model:
    def __init__(self, num_features, num_s):
        self.x = tf.placeholder(tf.float, shape=[None, input_size])
        self.s = tf.placeholder(tf.float, shape=[None, num_s])

        self.in_layer = x
        self.h_layer = tf.layers.dense(in_layer, 10)
        self.out = tf.layers.dense(h_layer, num_s)

        self.loss = tf.losses.sigmoid_cross_entropy(self.s, self.out)




file = sys.argv[1]

df = pandas.read_csv(file)
y_columns = df.filter(regex=("y.*")).values
s_columns = df.filter(regex=("s.*")).values
h_columns = df.filter(regex=("h.*")).values

h_train, h_test, s_train, s_test, y_train, y_test = ms.train_test_split(
    h_columns, s_columns, y_columns, random_state=RANDOM_SEED)

init = tf.global_variables_initializer()
session = tf.Session()

session.run(init)

for epoch in len(range(EPOCHS)):
    batch_pos = 0
    while batch_pos < len(h_train):
        batch = h_train[batch_pos:(batch_pos+BATCH_SIZE)]
        batch_pos += BATCH_SIZE






svc_y.fit(h_train, y_train)
pred_y_train = svc_y.predict(h_train)
pred_y_test = svc_y.predict(h_test)

acc_y_train = sum(pred_y_train == y_train)/float(len(y_train))
acc_y_test = sum(pred_y_test == y_test)/float(len(y_test))

svc_s = svm.SVC(random_state=RANDOM_SEED)

svc_s.fit(h_train, s_train)

pred_s_train = svc_s.predict(h_train)
pred_s_test = svc_s.predict(h_test)

acc_s_train = sum(pred_s_train == s_train)/float(len(s_train))
acc_s_test = sum(pred_s_test == s_test)/float(len(s_test))

return (acc_s_train, acc_y_train, acc_s_test, acc_y_test)


