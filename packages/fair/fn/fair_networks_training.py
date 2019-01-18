from __future__ import print_function
import tensorflow as tf
import numpy as np
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

        self.train_noise = np.random.uniform(size=self.train_xs.shape)
        self.test_noise = np.random.uniform(size=self.test_xs.shape)

        self.trainset = self.dataset.train_dataset().batch(self.options.batch_size).shuffle(1000)
        self.trainset_it = self.trainset.make_initializable_iterator()
        self.trainset_next = self.trainset_it.get_next()

        self.saver = saver

        self.writer = writer

        self.s_variables = [var for varlist in self.model.s_variables for var in varlist]
        self.init_s_vars = tf.variables_initializer(self.s_variables, name="init_s_vars")


        self.y_variables = [var for varlist in self.model.y_variables for var in varlist]
        self.init_y_vars = tf.variables_initializer(self.y_variables, name="init_y_vars")


    def run_train_s(self, xs, s):
        if self.options.fairness_importance == 0:
            return
            
        self.session.run(self.init_s_vars)

        for _ in range(self.options.schedule.sub_nets_num_it):

            batch_pos = 0
            while batch_pos < len(xs):
                batch_x = xs[batch_pos:(batch_pos+self.options.batch_size)]
                batch_s =  s[batch_pos:(batch_pos+self.options.batch_size)]
                noise = self.train_noise[batch_pos:(batch_pos + len(batch_x))]
                batch_pos += self.options.batch_size

                self.session.run(self.model.s_train_step, feed_dict={
                    self.model.x: batch_x, 
                    self.model.s: batch_s,
                    self.model.noise: noise })

    def run_epoch_new_approach(self):
        dataset_size = len(self.train_xs)
        batch = 0
        noise_batch_start = 0
        tot_batches = dataset_size / self.options.batch_size
        self.session.run(self.trainset_it.initializer)

        epoch = self.session.run(self.model.epoch)

        while True:
            try:
                xs, ys, s = self.session.run(self.trainset_next)
                noise = self.train_noise[noise_batch_start:(noise_batch_start+len(xs))]
                noise_batch_start += len(xs)

                self.run_train_s(self.train_xs, self.train_s)
                self.session.run(self.model.y_train_step, feed_dict = { self.model.x:xs, self.model.y:ys, self.model.noise:noise })
                self.session.run(self.model.h_train_step, feed_dict = { self.model.x:xs, self.model.y:ys, self.model.s:s, self.model.noise:noise })

                batch += 1
                perc_complete = (float(batch) / tot_batches) * 100
                print("\rProcessing epoch %d batch:%d/%d (%2.2f%%)" %
                      (epoch, batch, tot_batches, perc_complete), end="")

                if tot_batches > 20 and batch % int(tot_batches / 20)  == 0:
                    print("\n")
                    self.run_train_s(self.train_xs, self.train_s)
                    self.log_stats(epoch)

            except tf.errors.OutOfRangeError:
                break

    def run_epoch_batched(self):
        dataset_size = len(self.train_xs)
        batch = 0
        noise_batch_start = 0
        tot_batches = dataset_size / self.options.batch_size
        self.session.run(self.trainset_it.initializer)

        epoch = self.session.run(self.model.epoch)

        while True:
            try:
                xs, ys, s = self.session.run(self.trainset_next)
                noise = self.train_noise[noise_batch_start:(noise_batch_start+len(xs))]
                noise_batch_start += len(xs)

                self.session.run(self.model.s_train_step, feed_dict = { 
                    self.model.x:xs, 
                    self.model.s:s, 
                    self.model.noise:noise })
                self.session.run(self.model.y_train_step, feed_dict = { 
                    self.model.x:xs, 
                    self.model.y:ys,
                    self.model.noise:noise })
                self.session.run(self.model.h_train_step, feed_dict = { 
                    self.model.x:xs, 
                    self.model.y:ys, 
                    self.model.s:s, 
                    self.model.noise:noise })

                if tot_batches > 20 and batch % int(tot_batches / 20)  == 0:
                    print("\n")
                    self.log_stats(epoch)

            except tf.errors.OutOfRangeError:
                break


    def training_loop(self):
        for _ in range(self.options.schedule.num_epochs):
            epoch = self.session.run(self.model.epoch)

            if self.options.batched:
                self.run_epoch_batched()
            else:
                self.run_epoch_new_approach()

            self.save_model(int(epoch[0]))

            self.updateTensorboardStats(epoch)

            if int(epoch[0]) % 10 == 0:
                self.log_losses(epoch)
                self.model.print_weight(self.session, 2)
                self.model.print_weight(self.session, 3)

            self.session.run(self.model.inc_epoch)

        self.save_model("final")

    def log_losses(self, epoch):
        nn_y_loss = self.session.run(self.model.y_loss, feed_dict = {
            self.model.x: self.test_xs, 
            self.model.y: self.test_ys,
            self.model.noise: self.test_noise
            })
        nn_s_loss = self.session.run(self.model.s_mean_loss, feed_dict = {
            self.model.x: self.test_xs, 
            self.model.s: self.test_s,
            self.model.noise: self.test_noise
            })
        nn_h_loss = self.session.run(self.model.h_loss, feed_dict = {
            self.model.x: self.test_xs, 
            self.model.s: self.test_s, 
            self.model.y: self.test_ys,
            self.model.noise: self.test_noise})

        print('Epoch {:4} y loss: {:07.6f} s loss: {:07.6f} h loss: {:07.6f}'.format(int(epoch[0]), nn_y_loss, nn_s_loss, nn_h_loss))

    def log_stats(self, epoch):
        self.run_train_s(self.train_xs, self.train_s)

        nn_y_accuracy = self.session.run(self.model.y_accuracy, feed_dict = {
            self.model.x: self.test_xs, 
            self.model.y: self.test_ys,
            self.model.noise: self.test_noise})
        nn_s_accuracy = self.session.run(self.model.s_accuracy, feed_dict = {
            self.model.x: self.test_xs, 
            self.model.s: self.test_s,
            self.model.noise: self.test_noise})

        print("Epoch %f y accuracy: %2.4f s accuracy: %2.4f" % (epoch[0], nn_y_accuracy, nn_s_accuracy))

    def updateTensorboardStats(self, epoch):
        stat_des = self.session.run(self.model.train_stats, feed_dict = { 
            self.model.x:self.train_xs, 
            self.model.y:self.train_ys, 
            self.model.s: self.train_s,
            self.model.noise: self.train_noise })

        self.writer.add_summary(stat_des, global_step = epoch)

        stat_des = self.session.run(self.model.test_stats, feed_dict = { 
            self.model.x:self.test_xs, 
            self.model.y:self.test_ys, 
            self.model.s: self.test_s,
            self.model.noise: self.test_noise })
        self.writer.add_summary(stat_des, global_step = epoch)


    def save_model(self, epoch):
        if epoch == "final" or self.options.save_at_epoch(epoch):
            self.saver.save(self.session, self.options.output_fname(epoch))
