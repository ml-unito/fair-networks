import sys
sys.path.append('code')

import tensorflow as tf
import time
import adult_dataset as ds

NUM_EPOCHS = 1000

# Building model

def build_model(layers, optimizer):
    x = tf.placeholder(tf.float32, shape=[None, 108], name="x")
    y = tf.placeholder(tf.float32, shape=[None, 2], name="y")
    is_training = tf.placeholder(tf.bool, None)
    in_layer = x

    for index,layer in enumerate(layers):
        num_nodes, activation, initializer = layer
        with tf.name_scope("layer-%d" % (index+1)):
            in_layer = tf.layers.dense(in_layer, num_nodes, activation=activation, kernel_initializer = initializer())

    with tf.name_scope("out"):
        out = tf.layers.dense(in_layer, 2, activation=tf.nn.sigmoid, kernel_initializer = tf.random_normal_initializer())

    with tf.name_scope("loss"):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=out))
        loss_stat = tf.summary.scalar("softmax_loss", loss)

    with tf.name_scope("accuracy"):
        correct_predictions = tf.cast(tf.equal(tf.argmax(out,1), tf.argmax(y,1)), "float")
        accuracy = tf.cast(tf.reduce_mean(correct_predictions), "float")
        accuracy_stat = tf.summary.scalar("Accuracy", accuracy)

    train_step = optimizer.minimize(loss)

    return (x,y,train_step,is_training,loss,accuracy)

def eval_loss_and_accuracy(session, loss, accuracy, xs, ys):
    loss_val = session.run(loss, feed_dict={x: xs, y: ys, is_training: False})
    accuracy_val = session.run(accuracy, feed_dict={x: xs, y: ys, is_training: False})
    return (loss_val, accuracy_val)

# --------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------

x,y,train_step,is_training,loss,accuracy = build_model([
    (10, tf.nn.sigmoid, tf.random_normal_initializer)],
    tf.train.GradientDescentOptimizer(3.0))

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

print("%6s|%6s|%10s|%10s|%10s|%10s" % ("epoch", "it", "acc train", "acc test", "loss train", "loss test"))

count = 0

for _ in range(NUM_EPOCHS):
    session.run(train_it.initializer)
    while True:
        try:
            count += 1
            xs, ys = session.run(train_next)
            session.run(train_step, feed_dict = { x:xs, y:ys, is_training: True } )
        except tf.errors.OutOfRangeError:
          break

    last_epoch += 1

    loss_train_val, accuracy_train_val = eval_loss_and_accuracy(session, loss, accuracy, train_xs, train_ys)

    loss_test_val, accuracy_test_val = eval_loss_and_accuracy(session, loss, accuracy, test_xs, test_ys)


    print("%6d|%6d|%2.8f|%2.8f|%2.8f|%2.8f" % (last_epoch, count, accuracy_train_val, accuracy_test_val, loss_train_val, loss_test_val))
