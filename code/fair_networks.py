import sys
import numpy as np
import tensorflow as tf
import time

sys.path.append('code')

from model import Model
from options import Options


# --------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------

opts = Options()
dataset = opts.dataset

optimizer = tf.train.AdagradOptimizer(1.0)

model = Model(opts, optimizer)

train_xs, train_ys, train_s = dataset.train_all_data()
test_xs, test_ys, test_s = dataset.test_all_data()

train_feed = { model.x:train_xs, model.y:train_ys, model.s:train_s }
test_feed = { model.x:test_xs, model.y:test_ys, model.s:test_s }

trainset = dataset.train_dataset().batch(100).shuffle(1000)
trainset_it = trainset.make_initializable_iterator()
trainset_next = trainset_it.get_next()

session = tf.Session()
saver = tf.train.Saver()

writer = tf.summary.FileWriter(logdir=opts.log_fname())

if opts.resume_learning:
    model_to_resume = opts.model_fname(opts.epoch_start)
    print("Restoring model: %s" % (model_to_resume))
    saver.restore(session, model_to_resume)
else:
    print("Initializing a new model")
    init = tf.global_variables_initializer()
    session.run(init)
    writer.add_graph(session.graph)

for epoch in opts.epochs:
    session.run(trainset_it.initializer)
    while True:
        try:
            xs, ys, s = session.run(trainset_next)

            # session.run(model.y_train_step, feed_dict = { model.x:xs, model.y:ys })
            session.run(model.s_train_step, feed_dict = { model.x:xs, model.s:s })

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

        # print("Errors:")
        # model.print_errors(session, train_feed, model.s_out, model.s)

    stat_des = session.run(model.train_stats, feed_dict = { model.x:train_xs, model.y:train_ys, model.s: train_s })
    writer.add_summary(stat_des, global_step = epoch)

    stat_des = session.run(model.test_stats, feed_dict = { model.x:test_xs, model.y:test_ys, model.s: test_s })
    writer.add_summary(stat_des, global_step = epoch)


saver.save(session, opts.model_fname(epoch+1))
