import sys
from sklearn import svm
from sklearn import tree
import numpy as np
import tensorflow as tf
import time

sys.path.append('code')

from model import build_model
import bank_marketing_dataset as ds
# import adult_dataset as ds

def eval_loss_and_accuracy(session, loss, accuracy, xs, ys):
    loss_val = session.run(loss, feed_dict={x: xs, y: ys})
    accuracy_val = session.run(accuracy, feed_dict={x: xs, y: ys})
    return (loss_val, accuracy_val)

def print_confusion_matrix(tp, tn, fp, fn):
    print("|        |predicted +|predicted -|")
    print("|:------:|----------:|----------:|")
    print("|actual +|%11d|%11d|" % (tp,fn))
    print("|actual -|%11d|%11d|" % (fp,tn))


# --------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------

# NUM_FEATURES = 92   # Adult
NUM_FEATURES = 51     # Bank
NUM_EPOCHS = 10000
HIDDEN_UNITS = 100

HIDDEN_LAYERS = [
    (HIDDEN_UNITS, tf.nn.sigmoid, tf.truncated_normal_initializer) # first layer
    ]
EXP_NAME = "adult_h%d" % HIDDEN_UNITS


dataset = ds.BankMarketingDataset()

optimizer = tf.train.AdagradOptimizer(1.0)
x, y, train_step, loss, accuracy, confusion_matrix, train_stats, test_stats = build_model(
    HIDDEN_LAYERS, optimizer, NUM_FEATURES )

# grads = optimizer.compute_gradients(loss)
# for index, grad in enumerate(grads):
#     tf.summary.histogram("{}-grad".format(grads[index][1].name), grads[index])

train_xs, train_ys = dataset.train_all_data()
test_xs, test_ys = dataset.test_all_data()

trainset = dataset.train_dataset().batch(100).shuffle(1000)
trainset_it = trainset.make_initializable_iterator()
trainset_next = trainset_it.get_next()


session = tf.Session()
init = tf.global_variables_initializer()
session.run(init)


writer = tf.summary.FileWriter(logdir='logdir/log_%s' % EXP_NAME, graph=session.graph)
saver = tf.train.Saver()

for epoch in range(NUM_EPOCHS):
    session.run(trainset_it.initializer)
    while True:
        try:
            xs, ys = session.run(trainset_next)
            session.run(train_step, feed_dict = { x:xs, y:ys } )
        except tf.errors.OutOfRangeError:
          break

    saver.save(session, "models/%s.ckpt" % EXP_NAME)

    stat_des = session.run(train_stats, feed_dict = { x:train_xs, y:train_ys })
    writer.add_summary(stat_des, global_step = epoch)

    stat_des = session.run(test_stats, feed_dict = { x:test_xs, y:test_ys })
    writer.add_summary(stat_des, global_step = epoch)


    loss_train_val, accuracy_train_val = eval_loss_and_accuracy(session, loss, accuracy, train_xs, train_ys)
    loss_test_val, accuracy_test_val = eval_loss_and_accuracy(session, loss, accuracy, test_xs, test_ys)

    (TP_val,TN_val,FP_val,FN_val) = session.run(confusion_matrix, feed_dict = {x:train_xs, y:train_ys})

    print("")
    print("%6d|%2.8f|%2.8f|%2.8f|%2.8f" % (epoch, accuracy_train_val, accuracy_test_val, loss_train_val, loss_test_val))
    print_confusion_matrix(TP_val,TN_val,FP_val,FN_val)
    print("")
