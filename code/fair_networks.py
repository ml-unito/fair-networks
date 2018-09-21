import sys
import numpy as np
import tensorflow as tf
import time
from termcolor import colored

sys.path.append('code')

from model import Model
from options import Options

SUB_NETS_NUM_IT = 10

def run_epoch_new_approach(session, model, trainset_next, init_y_vars, init_s_vars):

    while True:
        try:
            xs, ys, s = session.run(trainset_next)

            session.run(init_s_vars)
            session.run(init_y_vars)

            for i in range(SUB_NETS_NUM_IT):
                session.run(model.y_train_step, feed_dict = { model.x:xs, model.y:ys })

            for i in range(SUB_NETS_NUM_IT):
                session.run(model.s_train_step, feed_dict = { model.x:xs, model.s:s })

            session.run(model.h_train_step, feed_dict = { model.x:xs, model.y:ys, model.s:s })

            session.run(init_s_vars)
            for i in range(SUB_NETS_NUM_IT):
                session.run(model.s_train_step, feed_dict = { model.x:xs, model.s:s })

        except tf.errors.OutOfRangeError:
          break

def training_loop():
    s_variables = [var for varlist in model.s_variables for var in varlist]
    init_s_vars = tf.variables_initializer(s_variables, name="init_s_vars")


    s_variables = [var for varlist in model.s_variables for var in varlist]
    init_y_vars = tf.variables_initializer(s_variables, name="init_y_vars")

    while True:
        step_type = opts.schedule.get_next()

        if step_type == None:
            break

        session.run(trainset_it.initializer)
        run_epoch_new_approach(session, model, trainset_next, init_y_vars, init_s_vars)


        if opts.save_at_epoch(session.run(model.epoch)):
            saver.save(session, opts.output_fname())

            print('\n--------------------------------------------------------------')
            print(colored("epoch: %d" % session.run(model.epoch), 'green', attrs=['bold']))
            model.print_loss_and_accuracy(session, train_feed_dict = train_feed, test_feed_dict = test_feed)

            print(colored("\nConfusion matrix -- Train:", attrs=['bold']))
            model.print_confusion_matrix(session, feed_dict = train_feed)

            print(colored("\nConfusion matrix -- Test:", attrs=['bold']))
            model.print_confusion_matrix(session, feed_dict = test_feed)

            # print("Errors:")
            # model.print_errors(session, train_feed, model.s_out, model.s)

        stat_des = session.run(model.train_stats, feed_dict = { model.x:train_xs, model.y:train_ys, model.s: train_s })
        writer.add_summary(stat_des, global_step = session.run(model.epoch))

        stat_des = session.run(model.test_stats, feed_dict = { model.x:test_xs, model.y:test_ys, model.s: test_s })
        writer.add_summary(stat_des, global_step = session.run(model.epoch))

        session.run(model.inc_epoch)

    saver.save(session, opts.output_fname())

def print_stats():
    print("Learning rate:")
    print(session.run(optimizer._learning_rate_tensor))

    print("loss and accuracy:")
    model.print_loss_and_accuracy(session, train_feed_dict = train_feed, test_feed_dict = test_feed)

    print(colored("\nConfusion matrix -- Train:", attrs=['bold']))
    model.print_confusion_matrix(session, feed_dict = train_feed)

    print(colored("\nConfusion matrix -- Test:", attrs=['bold']))
    model.print_confusion_matrix(session, feed_dict = test_feed)

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
    model_to_resume = opts.input_fname()
    print(colored("Restoring model: %s" % (model_to_resume), 'yellow'))
    saver.restore(session, model_to_resume)
else:
    print(colored("Initializing a new model", 'yellow'))
    init = tf.global_variables_initializer()
    session.run(init)
    writer.add_graph(session.graph)


if not opts.eval_stats:
    training_loop()
else:
    print_stats()
