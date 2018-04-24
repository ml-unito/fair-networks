import sys
from sklearn import svm
from sklearn import tree
import numpy as np
import tensorflow as tf
import time

sys.path.append('code')

from model import build_model, print_confusion_matrix, print_loss_and_accuracy
import bank_marketing_dataset as ds



if len(sys.argv) < 3:
    print("Usage: %s <num_hidden_units> <epoch>" % sys.argv[0])
    sys.exit(1)

NUM_FEATURES = 51     # Bank
NUM_EPOCHS = 10000
HIDDEN_UNITS = int(sys.argv[1])
EPOCH = int(sys.argv[2])

HIDDEN_LAYERS = [
    (HIDDEN_UNITS, tf.nn.sigmoid, tf.truncated_normal_initializer) # first layer
    ]
EXP_NAME = "bank_h%d" % HIDDEN_UNITS


dataset = ds.BankMarketingDataset()

optimizer = tf.train.AdagradOptimizer(1.0)
x, y, train_step, loss, accuracy, confusion_matrix, train_stats, test_stats = build_model(
    HIDDEN_LAYERS, optimizer, NUM_FEATURES )


train_xs, train_ys = dataset.train_all_data()
test_xs, test_ys = dataset.test_all_data()

session = tf.Session()

saver = tf.train.Saver()
saver.restore(session, 'models/%s-epoch-%d.ckpt' % (EXP_NAME, EPOCH))

train_feed = { x:train_xs, y:train_ys }
test_feed = { x:test_xs, y:test_ys }

print_loss_and_accuracy(session, loss, accuracy, train_feed_dict = train_feed, test_feed_dict = test_feed)

print("Confusion matrix -- Train")
print_confusion_matrix(session, confusion_matrix, feed_dict = train_feed)

print("Confusion matrix -- Test")
print_confusion_matrix(session, confusion_matrix, feed_dict = test_feed)
