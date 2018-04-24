import sys
from sklearn import svm
from sklearn import tree
import numpy as np
import tensorflow as tf
import time

sys.path.append('code')

from model import build_model, print_confusion_matrix, eval_loss_and_accuracy
import bank_marketing_dataset as ds



if len(sys.argv) < 2:
    print("Usage: %s <num_hidden_units>" % sys.argv[0])
    sys.exit(1)

NUM_FEATURES = 51     # Bank
NUM_EPOCHS = 10000
HIDDEN_UNITS = sys.argv[1]

HIDDEN_LAYERS = [
    (HIDDEN_UNITS, tf.nn.sigmoid, tf.truncated_normal_initializer) # first layer
    ]
EXP_NAME = "adult_h%d" % HIDDEN_UNITS


dataset = ds.BankMarketingDataset()

optimizer = tf.train.AdagradOptimizer(1.0)
x, y, train_step, loss, accuracy, confusion_matrix, train_stats, test_stats = build_model(
    HIDDEN_LAYERS, optimizer, NUM_FEATURES )


train_xs, train_ys = dataset.train_all_data()
test_xs, test_ys = dataset.test_all_data()

session = tf.Session()

saver = tf.train.Saver()
saver.restore(session, 'models/%s.ckpt' % EXP_NAME)


loss_train_val, accuracy_train_val = eval_loss_and_accuracy(session, loss, accuracy, feed_dict = { x:train_xs, y:train_ys} )
loss_test_val, accuracy_test_val = eval_loss_and_accuracy(session, loss, accuracy, feed_dict = {x:test_xs, y:test_ys})


print("")
print("%2.8f|%2.8f|%2.8f|%2.8f" % (accuracy_train_val, accuracy_test_val, loss_train_val, loss_test_val))

print("Confusion Matrix Train")
(TP_val,TN_val,FP_val,FN_val) = session.run(confusion_matrix, feed_dict = {x:train_xs, y:train_ys})

print_confusion_matrix(TP_val,TN_val,FP_val,FN_val)

print("Confusione Matrix Test")
(TP_val,TN_val,FP_val,FN_val) = session.run(confusion_matrix, feed_dict = {x:test_xs, y:test_ys})

print_confusion_matrix(TP_val,TN_val,FP_val,FN_val)
