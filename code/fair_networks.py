import sys
sys.path.append('code')

import tensorflow as tf
import time
import adult_dataset as ds

NUM_EPOCHS = 1000


# Building model

x = tf.placeholder(tf.float32, shape=[None, 108], name="x")
y = tf.placeholder(tf.float32, shape=[None, 2], name="y")
is_training = tf.placeholder(tf.bool, None)

with tf.name_scope("layer-1"):
    l1 = tf.layers.dense(x, 25, activation=tf.nn.sigmoid, kernel_initializer = tf.random_normal_initializer())
    # l1 = tf.layers.dropout(l1, rate = 0.5, training = is_training)

with tf.name_scope("layer-2"):
    l2 = tf.layers.dense(l1, 25, activation=tf.nn.sigmoid, kernel_initializer = tf.random_normal_initializer())
    # l2 = tf.layers.dropout(l2, rate = 0.5, training = is_training)

with tf.name_scope("out"):
    out = tf.layers.dense(l2, 2, activation=tf.nn.sigmoid, kernel_initializer = tf.random_normal_initializer())

with tf.name_scope("loss"):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=out))
    loss_stat = tf.summary.scalar("softmax_loss", loss)

with tf.name_scope("accuracy"):
    correct_predictions = tf.cast(tf.equal(tf.argmax(out,1), tf.argmax(y,1)), "float")
    class_accuracy_op = tf.cast(tf.reduce_mean(correct_predictions), "float")
    accuracy_stat = tf.summary.scalar("Accuracy", class_accuracy_op)

train_step = tf.train.GradientDescentOptimizer(3.0).minimize(loss)

dataset = ds.AdultDataset()

train_xs, train_ys = dataset.train_all_data()
test_xs, test_ys = dataset.test_all_data()


trainset = dataset.train_dataset().batch(100)
train_it = trainset.make_initializable_iterator()
train_next = train_it.get_next()

last_epoch = 0

session = tf.Session()
init = tf.global_variables_initializer()
session.run(init)

for _ in range(NUM_EPOCHS):
    session.run(train_it.initializer)
    while True:
        try:
            xs, ys = session.run(train_next)
            session.run(train_step, feed_dict = { x:xs, y:ys, is_training: True } )
        except tf.errors.OutOfRangeError:
          break

    last_epoch += 1

    loss_val = session.run(loss, feed_dict={x: test_xs, y: test_ys, is_training: False})
    accuracy_val = session.run(class_accuracy_op, feed_dict={x: test_xs, y: test_ys, is_training: False})
    print(" test results -- epoch: %3d\taccuracy: %2.5f\tloss: %2.5f" % (last_epoch, accuracy_val, loss_val))

    loss_val = session.run(loss, feed_dict={x: train_xs, y: train_ys, is_training: False})
    accuracy_val = session.run(class_accuracy_op, feed_dict={x: train_xs, y: train_ys, is_training: False})
    print("train results -- epoch: %3d\taccuracy: %2.5f\tloss: %2.5f" % (last_epoch, accuracy_val, loss_val))
