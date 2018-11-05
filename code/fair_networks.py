import sys
import numpy as np
import tensorflow as tf
import time
import pandas
import sklearn.svm as svm
from termcolor import colored

sys.path.append('code')

from model import Model
from options import Options

def svc_results_stats(h_train, h_test):
    svc_y = svm.SVC(gamma="scale")
    svc_y.fit(h_train, train_ys[:,1])
    y_pred = svc_y.predict(h_test)


    y_accuracy = 1.0 - np.mean(test_ys[:,1] != y_pred)

    svc_s = svm.SVC(gamma="scale")
    s_train = np.argmax(train_s, axis=1)
    s_test = np.argmax(test_s, axis=1)
    svc_s.fit(h_train, s_train)
    s_pred = svc_s.predict(h_test)

    s_accuracy = 1.0 - np.mean(s_test != s_pred)

    print("SVC -- y accuracy: %2.4f   s accuracy: %2.4f" % (y_accuracy, s_accuracy))
    return (y_accuracy, s_accuracy)


def train_s_and_y(session, model, init_y_vars, init_s_vars, xs, ys, s):
    session.run(init_s_vars)
    session.run(init_y_vars)

    for i in range(opts.schedule.sub_nets_num_it):
        session.run(model.y_train_step, feed_dict = { model.x:xs, model.y:ys })

    for i in range(opts.schedule.sub_nets_num_it):
        session.run(model.s_train_step, feed_dict = { model.x:xs, model.s:s })
        # print("NN sub: s accuracy: %2.4f" % (session.run(model.s_accuracy, feed_dict={model.x:xs, model.s:s})))

    # print("----")

def run_epoch_new_approach(session, model, trainset_next, init_y_vars, init_s_vars):

    while True:
        try:
            xs, ys, s = session.run(trainset_next)

            train_s_and_y(session, model, init_y_vars, init_s_vars, xs, ys, s)

            session.run(model.h_train_step, feed_dict = { model.x:xs, model.y:ys, model.s:s })

        except tf.errors.OutOfRangeError:
          break

def training_loop():
    s_variables = [var for varlist in model.s_variables for var in varlist]
    init_s_vars = tf.variables_initializer(s_variables, name="init_s_vars")


    s_variables = [var for varlist in model.s_variables for var in varlist]
    init_y_vars = tf.variables_initializer(s_variables, name="init_y_vars")

    for _ in range(opts.schedule.num_epochs):
        epoch = session.run(model.epoch)
        print("Running epoch number: %d" % epoch)

        session.run(trainset_it.initializer)
        run_epoch_new_approach(session, model, trainset_next, init_y_vars, init_s_vars)

        # retrains the s layer so to be sure to have the best possible prediction about its
        # performances
        train_s_and_y(session, model, init_y_vars, init_s_vars, train_xs, train_ys, train_s)

        if opts.save_at_epoch(epoch):
            saver.save(session, opts.output_fname())

            print('\n--------------------------------------------------------------')
            print(colored("epoch: %d" % epoch, 'green', attrs=['bold']))
            model.print_loss_and_accuracy(session, train_feed_dict = train_feed, test_feed_dict = test_feed)

            print(colored("\nConfusion matrix -- Train:", attrs=['bold']))
            model.print_confusion_matrix(session, feed_dict = train_feed)

            print(colored("\nConfusion matrix -- Test:", attrs=['bold']))
            model.print_confusion_matrix(session, feed_dict = test_feed)

            # print("Errors:")
            # model.print_errors(session, train_feed, model.s_out, model.s)


        stat_des = session.run(model.train_stats, feed_dict = { model.x:train_xs, model.y:train_ys, model.s: train_s })
        writer.add_summary(stat_des, global_step = epoch)

        stat_des = session.run(model.test_stats, feed_dict = { model.x:test_xs, model.y:test_ys, model.s: test_s })
        writer.add_summary(stat_des, global_step = epoch)

        nn_y_accuracy = session.run(model.y_accuracy, feed_dict = {model.x: test_xs, model.y: test_ys})
        nn_s_accuracy = session.run(model.s_accuracy, feed_dict = {model.x: test_xs, model.s: test_s})

        print("NN -- y accuracy: %s    s accuracy: %s" % (nn_y_accuracy, nn_s_accuracy))


        # SVC summaries
        h_train = session.run(model.model_last_hidden_layer, feed_dict={model.x: train_xs})
        h_test = session.run(model.model_last_hidden_layer, feed_dict={model.x: test_xs})

        y_accuracy, s_accuracy = svc_results_stats(h_train, h_test)
        y_svc_stat, s_svc_stat = session.run((model.y_svc_accuracy_stat, model.s_svc_accuracy_stat), feed_dict={
            model.y_svc_accuracy:y_accuracy, model.s_svc_accuracy:s_accuracy })
        writer.add_summary(y_svc_stat, global_step = epoch)
        writer.add_summary(s_svc_stat, global_step = epoch)

        # Changing epoch
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

def print_processed_data():
    model_data_representation = session.run(model.model_last_hidden_layer, feed_dict={model.x:train_xs, model.y:train_ys, model.s:train_s})

    result = np.hstack((model_data_representation, train_s, train_ys))
    h_header = ["h_"+str(index) for index in range(len(model_data_representation[0]))]
    s_header = ["s_"+str(index) for index in range(len(train_s[0]))]
    y_header = ["y_"+str(index) for index in range(len(train_ys[0]))]

    print(colored("Saving data representations onto %s" % opts.eval_data_path, "green"))
    pandas.DataFrame(result, columns=h_header + s_header + y_header).to_csv(opts.eval_data_path, index=False)



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


if opts.eval_stats:
    print_stats()
elif opts.eval_data_path != None:
    print_processed_data()
else:
    training_loop()
