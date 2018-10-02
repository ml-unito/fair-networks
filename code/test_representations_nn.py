import pandas
import sklearn.model_selection as ms
import numpy as np
import sys
import tensorflow as tf

BATCH_SIZE=100

# sys.argv[1]='synth_h10_s5_y5_data.csv'

df = pandas.read_csv(sys.argv[1])
y_columns = df.filter(regex=("y.*")).as_matrix()
s_columns = df.filter(regex=("s.*")).as_matrix()
h_columns = df.filter(regex=("h.*")).as_matrix()

h_train, h_test, s_train, s_test, y_train, y_test = ms.train_test_split(h_columns, s_columns, y_columns)


xs = tf.placeholder(tf.float32, shape=[None, h_columns.shape[1]])
ys = tf.placeholder(tf.float32, shape=[None, s_columns.shape[1]])

l1 = tf.layers.dense(xs, 5)
l2 = tf.layers.dense(l1, 5)

s_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=l2, labels=ys))

optimizer = tf.train.AdagradOptimizer(1.0)
train_step = optimizer.minimize(s_loss)

session = tf.Session()
init = tf.global_variables_initializer()

session.run(init)

for i in range(100):
    session.run(train_step, feed_dict = { xs: h_train[:BATCH_SIZE,:], ys: s_train[:BATCH_SIZE,:] })
    print(session.run(s_loss, feed_dict = {xs: h_train[:BATCH_SIZE,:], ys: s_train[:BATCH_SIZE,:]}))

accuracy = tf.reduce_sum(tf.cast(tf.equal(tf.argmax(l2, axis=1), tf.argmax(s_train, axis=1)), tf.int32)) / h_train.shape[0]

print(session.run(accuracy , feed_dict = {xs: h_train}))


# print("s accuracy:" + str(1.0 - np.mean(np.abs(s_test - s_pred))))
