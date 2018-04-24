import sys
from sklearn import svm
from sklearn import tree
import numpy as np
import tensorflow as tf
import time

sys.path.append('code')

from model import build_model, print_confusion_matrix, print_loss_and_accuracy
import bank_marketing_dataset as ds
# import adult_dataset as ds


# --------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------

if len(sys.argv) < 2:
    print("Usage: %s <num_hidden_units> [epoch_start:epoch_end]" % sys.argv[0])
    sys.exit(1)

resume_learning = len(syss.argv) == 3

# NUM_FEATURES = 92   # Adult
NUM_FEATURES = 51     # Bank

if resume_learning:
    epoch_start, epoch_end = sys.argv[2].split(':')
    EPOCHS = range(epoch_start, epoch_end)
else
    EPOCHS = range(0,10000)

HIDDEN_UNITS = int(sys.argv[1])
SAVE_FREQUENCY = 1000

HIDDEN_LAYERS = [
    (HIDDEN_UNITS, tf.nn.sigmoid, tf.truncated_normal_initializer) # first layer
    ]
EXP_NAME = "bank_h%d" % HIDDEN_UNITS


dataset = ds.BankMarketingDataset()

optimizer = tf.train.AdagradOptimizer(1.0)
x, y, train_step, loss, accuracy, confusion_matrix, train_stats, test_stats = build_model(
    HIDDEN_LAYERS, optimizer, NUM_FEATURES )

# grads = optimizer.compute_gradients(loss)
# for index, grad in enumerate(grads):
#     tf.summary.histogram("{}-grad".format(grads[index][1].name), grads[index])

train_xs, train_ys = dataset.train_all_data()
test_xs, test_ys = dataset.test_all_data()

train_feed = { x:train_xs, y:train_ys }
test_feed = { x:test_xs, y:test_ys }

trainset = dataset.train_dataset().batch(100).shuffle(1000)
trainset_it = trainset.make_initializable_iterator()
trainset_next = trainset_it.get_next()


session = tf.Session()
saver = tf.train.Saver()

if resume_learning:
    saver.restore(saver, "models/%s-epoch-%d.ckpt" % (EXP_NAME, epoch_start))
else:
    init = tf.global_variables_initializer()
    session.run(init)


writer = tf.summary.FileWriter(logdir='logdir/log_%s' % EXP_NAME, graph=session.graph)

for epoch in EPOCHS:
    session.run(trainset_it.initializer)
    while True:
        try:
            xs, ys = session.run(trainset_next)
            session.run(train_step, feed_dict = { x:xs, y:ys } )
        except tf.errors.OutOfRangeError:
          break

    if (epoch < SAVE_FREQUENCY and epoch % (SAVE_FREQUENCY/10) == 0) or epoch % SAVE_FREQUENCY == 0:
        saver.save(session, "models/%s-epoch-%d.ckpt" % (EXP_NAME, epoch))

        print_loss_and_accuracy(session, loss, accuracy, train_feed_dict = train_feed, test_feed_dict = test_feed)

        print("Confusion matrix -- Train")
        print_confusion_matrix(session, confusion_matrix, feed_dict = train_feed)

        print("Confusion matrix -- Test")
        print_confusion_matrix(session, confusion_matrix, feed_dict = test_feed)

    stat_des = session.run(train_stats, feed_dict = { x:train_xs, y:train_ys })
    writer.add_summary(stat_des, global_step = epoch)

    stat_des = session.run(test_stats, feed_dict = { x:test_xs, y:test_ys })
    writer.add_summary(stat_des, global_step = epoch)


saver.save(session, "models/%s-epoch-%d.ckpt" % (EXP_NAME, epoch+1))
