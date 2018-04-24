import sys
from sklearn import svm
from sklearn import tree
import numpy as np
import tensorflow as tf
import time

sys.path.append('code')

from model import Model
from options import Options
# import adult_dataset as ds


# --------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------

opts = Options().parse(sys.argv)
dataset = opts.dataset

optimizer = tf.train.AdagradOptimizer(1.0)
model = Model(opts.hidden_layers, optimizer, opts.num_features )

# grads = optimizer.compute_gradients(loss)
# for index, grad in enumerate(grads):
#     tf.summary.histogram("{}-grad".format(grads[index][1].name), grads[index])

train_xs, train_ys = dataset.train_all_data()
test_xs, test_ys = dataset.test_all_data()

train_feed = { model.x:train_xs, model.y:train_ys }
test_feed = { model.x:test_xs, model.y:test_ys }

trainset = dataset.train_dataset().batch(100).shuffle(1000)
trainset_it = trainset.make_initializable_iterator()
trainset_next = trainset_it.get_next()

session = tf.Session()
saver = tf.train.Saver()

if opts.resume_learning:
    saver.restore(session, opts.model_fname(opts.epoch_start))
else:
    init = tf.global_variables_initializer()
    session.run(init)


writer = tf.summary.FileWriter(logdir=opts.log_fname(), graph=session.graph)

for epoch in opts.epochs:
    session.run(trainset_it.initializer)
    while True:
        try:
            xs, ys = session.run(trainset_next)
            session.run(model.train_step, feed_dict = { model.x:xs, model.y:ys }  )
        except tf.errors.OutOfRangeError:
          break

    if opts.save_at_epoch(epoch):
        saver.save(session, opts.model_fname(epoch))

        print("epoch: %d" % epoch)
        model.print_loss_and_accuracy(session, train_feed_dict = train_feed, test_feed_dict = test_feed)

        print("Confusion matrix -- Train")
        model.print_confusion_matrix(session, feed_dict = train_feed)

        print("Confusion matrix -- Test")
        model.print_confusion_matrix(session, feed_dict = test_feed)

    stat_des = session.run(model.train_stats, feed_dict = { model.x:train_xs, model.y:train_ys })
    writer.add_summary(stat_des, global_step = epoch)

    stat_des = session.run(model.test_stats, feed_dict = { model.x:test_xs, model.y:test_ys })
    writer.add_summary(stat_des, global_step = epoch)


saver.save(session, opts.model_fname(epoch+1))
