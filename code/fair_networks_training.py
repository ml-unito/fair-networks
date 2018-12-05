import tensorflow as tf
from termcolor import colored



class FairNetworksTraining:
    def __init__(self, options, session, model, saver, writer):
        self.options = options
        self.dataset = options.dataset
        self.session = session
        self.model = model

        self.train_xs, self.train_ys, self.train_s = self.dataset.train_all_data()
        self.test_xs, self.test_ys, self.test_s = self.dataset.test_all_data()

        self.train_feed = {model.x: self.train_xs, model.y: self.train_ys, model.s: self.train_s}
        self.test_feed = {model.x: self.test_xs, model.y: self.test_ys, model.s: self.test_s}

        self.trainset = self.dataset.train_dataset().batch(self.options.batch_size).shuffle(1000)
        self.trainset_it = self.trainset.make_initializable_iterator()
        self.trainset_next = self.trainset_it.get_next()

        self.saver = saver

        self.writer = writer

        self.s_variables = [var for varlist in self.model.s_variables for var in varlist]
        self.init_s_vars = tf.variables_initializer(self.s_variables, name="init_s_vars")


        self.y_variables = [var for varlist in self.model.y_variables for var in varlist]
        self.init_y_vars = tf.variables_initializer(self.y_variables, name="init_y_vars")

    # def svc_results_stats(h_train, h_test):
    #     svc_y = svm.SVC()
    #     svc_y.fit(h_train, train_ys[:,1])
    #     y_pred = svc_y.predict(h_test)


    #     y_accuracy = 1.0 - np.mean(test_ys[:,1] != y_pred)

    #     svc_s = svm.SVC()
    #     s_train = np.argmax(train_s, axis=1)
    #     s_test = np.argmax(test_s, axis=1)
    #     svc_s.fit(h_train, s_train)
    #     s_pred = svc_s.predict(h_test)

    #     s_accuracy = 1.0 - np.mean(s_test != s_pred)

    #     print("SVC -- y accuracy: %2.4f   s accuracy: %2.4f" % (y_accuracy, s_accuracy))
    #     return (y_accuracy, s_accuracy)


    def run_train_s(self, xs, ys, s):
        self.session.run(self.init_s_vars)

        for _ in range(self.options.schedule.sub_nets_num_it):
            self.session.run(self.model.s_train_step, feed_dict = { self.model.x:xs, self.model.s:s })

    def run_epoch_new_approach(self):

        while True:
            try:
                xs, ys, s = self.session.run(self.trainset_next)

                self.run_train_s(xs, ys, s)

                self.session.run(self.model.y_train_step, feed_dict = { self.model.x:xs, self.model.y:ys })
                self.session.run(self.model.h_train_step, feed_dict = { self.model.x:xs, self.model.y:ys, self.model.s:s })

            except tf.errors.OutOfRangeError:
                break

    def training_loop(self):
        for _ in range(self.options.schedule.num_epochs):
            epoch = self.session.run(self.model.epoch)
            print("Running epoch number: %d" % epoch)

            self.session.run(self.trainset_it.initializer)
            self.run_epoch_new_approach()

            # retrains the s layer so to be sure to have the best possible prediction about its
            # performances
            self.run_train_s(self.train_xs, self.train_ys, self.train_s)

            if self.options.save_at_epoch(epoch):
                self.saver.save(self.session, self.options.output_fname())

                print('\n--------------------------------------------------------------')
                print(colored("epoch: %d" % epoch, 'green', attrs=['bold']))
                self.model.print_loss_and_accuracy(self.session, train_feed_dict = self.train_feed, test_feed_dict = self.test_feed)

                print(colored("\nConfusion matrix -- Train:", attrs=['bold']))
                self.model.print_confusion_matrix(self.session, feed_dict = self.train_feed)

                print(colored("\nConfusion matrix -- Test:", attrs=['bold']))
                self.model.print_confusion_matrix(self.session, feed_dict = self.test_feed)

                # print("Errors:")
                # self.model.print_errors(self.session, train_feed, self.model.s_out, self.model.s)


            stat_des = self.session.run(self.model.train_stats, feed_dict = { self.model.x:self.train_xs, self.model.y:self.train_ys, self.model.s: self.train_s })
            self.writer.add_summary(stat_des, global_step = epoch)

            stat_des = self.session.run(self.model.test_stats, feed_dict = { self.model.x:self.test_xs, self.model.y:self.test_ys, self.model.s: self.test_s })
            self.writer.add_summary(stat_des, global_step = epoch)

            nn_y_accuracy = self.session.run(self.model.y_accuracy, feed_dict = {self.model.x: self.test_xs, self.model.y: self.test_ys})
            nn_s_accuracy = self.session.run(self.model.s_accuracy, feed_dict = {self.model.x: self.test_xs, self.model.s: self.test_s})

            print("NN -- y accuracy: %s    s accuracy: %s" % (nn_y_accuracy, nn_s_accuracy))


            # # SVC summaries
            # if (epoch + 1) % 10 == 0:
            #     h_train = self.session.run(self.model.self.model_last_hidden_layer, feed_dict={self.model.x: train_xs})
            #     h_test = self.session.run(self.model.self.model_last_hidden_layer, feed_dict={self.model.x: test_xs})
            #
            #     y_accuracy, s_accuracy = svc_results_stats(h_train, h_test)
            #     y_svc_stat, s_svc_stat = self.session.run((self.model.y_svc_accuracy_stat, self.model.s_svc_accuracy_stat), feed_dict={
            #         self.model.y_svc_accuracy:y_accuracy, self.model.s_svc_accuracy:s_accuracy })
            #     self.writer.add_summary(y_svc_stat, global_step = epoch)
            #     self.writer.add_summary(s_svc_stat, global_step = epoch)

            # Changing epoch
            self.session.run(self.model.inc_epoch)

        self.saver.save(self.session, self.options.output_fname())