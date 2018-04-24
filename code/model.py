import tensorflow as tf
import time

# Building model

def build_model(layers, optimizer, num_features):
    x = tf.placeholder(tf.float32, shape=[None, num_features], name="x")
    y = tf.placeholder(tf.float32, shape=[None, 2], name="y")
    in_layer = x

    for index,layer in enumerate(layers):
        num_nodes, activation, initializer = layer
        with tf.name_scope("layer-%d" % (index+1)):
            in_layer = tf.layers.dense(in_layer, num_nodes, activation=activation, kernel_initializer = initializer())

    with tf.name_scope("out"):
        out = tf.layers.dense(in_layer, 2, activation=tf.nn.sigmoid, kernel_initializer = tf.truncated_normal_initializer())

    with tf.name_scope("loss"):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=out))
        train_loss_stat = tf.summary.scalar("train_softmax_loss", loss)
        test_loss_stat = tf.summary.scalar("test_softmax_loss", loss)

    with tf.name_scope("accuracy"):
        correct_predictions = tf.cast(tf.equal(tf.argmax(out,1), tf.argmax(y,1)), "float")
        accuracy = tf.cast(tf.reduce_mean(correct_predictions), "float")
        train_accuracy_stat = tf.summary.scalar("train_accuracy", accuracy)
        test_accuracy_stat = tf.summary.scalar("test_accuracy", accuracy)


    with tf.name_scope("confusion_matrix"):
        predicted = tf.argmax(out, 1)
        actual = tf.argmax(y,1)
        TP = tf.count_nonzero(predicted * actual)
        TN = tf.count_nonzero((predicted - 1) * (actual - 1))
        FP = tf.count_nonzero(predicted * (actual - 1))
        FN = tf.count_nonzero((predicted - 1) * actual)
        confusion_matrix = (TP,TN,FP,FN)

    train_step = optimizer.minimize(loss)
    train_stats = tf.summary.merge([train_loss_stat, train_accuracy_stat])
    test_stats = tf.summary.merge([test_loss_stat, test_accuracy_stat])

    return ( x, y, train_step, loss, accuracy, confusion_matrix, train_stats, test_stats)


def eval_loss_and_accuracy(session, loss, accuracy, feed_dict):
    loss_val = session.run(loss, feed_dict=feed_dict)
    accuracy_val = session.run(accuracy, feed_dict=feed_dict)
    return (loss_val, accuracy_val)

def print_confusion_matrix(tp, tn, fp, fn):
    print("|        |predicted +|predicted -|")
    print("|:------:|----------:|----------:|")
    print("|actual +|%11d|%11d|" % (tp,fn))
    print("|actual -|%11d|%11d|" % (fp,tn))
