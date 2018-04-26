import sys
from sklearn import svm
from sklearn import tree
import numpy as np
import tensorflow as tf
import time

sys.path.append('code')

from model import Model
from options import Options
import bank_marketing_dataset as ds


opts = Options().parse(sys.argv)
dataset = opts.dataset

optimizer = tf.train.AdagradOptimizer(1.0)
model = Model(opts.hidden_layers, optimizer, opts.num_features )

train_xs, train_ys = dataset.train_all_data()
test_xs, test_ys = dataset.test_all_data()

session = tf.Session()

saver = tf.train.Saver()
saver.restore(session, opts.model_fname(opts.epoch_end))

train_feed = { model.x:train_xs, model.y:train_ys }
test_feed = { model.x:test_xs, model.y:test_ys }

model.print_loss_and_accuracy(session, train_feed_dict = train_feed, test_feed_dict = test_feed)

print("Confusion matrix -- Train")
model.print_confusion_matrix(session, feed_dict = train_feed)

print("Confusion matrix -- Test")
model.print_confusion_matrix(session, feed_dict = test_feed)
