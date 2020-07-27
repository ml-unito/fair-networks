from __future__ import print_function
import tensorflow.compat.v1 as tf
import numpy as np
from termcolor import colored
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import logging


class FairNetworksTraining:
    def __init__(self, options, session, model, saver, writer):
        self.options = options
        self.dataset = options.dataset
        self.session = session
        self.model = model

        self.train_xs, self.train_ys, self.train_s = self.dataset.train_all_data()
        self.val_xs, self.val_ys, self.val_s = self.dataset.val_all_data()

        self.train_feed = {model.x: self.train_xs, model.y: self.train_ys, model.s: self.train_s}
        self.val_feed = {model.x: self.val_xs, model.y: self.val_ys, model.s: self.val_s}

        self.train_noise = np.random.uniform(size=self.train_xs.shape)
        self.val_noise = np.random.uniform(size=self.val_xs.shape)

        self.trainset = self.dataset.train_dataset().batch(self.options.batch_size).shuffle(1000)
        self.trainset_it = self.trainset.make_initializable_iterator()
        self.trainset_next = self.trainset_it.get_next()

        self.saver = saver

        self.writer = writer

        self.s_variables = [var for varlist in self.model.s_variables for var in varlist]
        self.init_s_vars = tf.variables_initializer(self.s_variables, name="init_s_vars")

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

                step_list = self.model.get_steps()

                for step in step_list:

                    self.session.run(step, feed_dict = {
                        self.model.x:xs,
                        self.model.s:s,
                        self.model.y:ys,
                        self.model.noise:noise })

            except tf.errors.OutOfRangeError:
                break


    def training_loop(self):
        epoch = int(self.session.run(self.model.epoch)[0])
        logging.info("Starting training loop from epoch: {}".format(epoch))

        while epoch < self.options.schedule.num_epochs:
            self.run_epoch_batched()
            self.updateTensorboardStats(epoch)

            if epoch % 10 == 0:
                self.log_losses(epoch)
                # self.log_stats_classifier(epoch)
                
                if self.options.verbose:
                    self.model.print_weight(self.session, 2)
                    self.model.print_weight(self.session, 3)

            self.session.run(self.model.inc_epoch)
            epoch = int(self.session.run(self.model.epoch)[0])
            self.save_model(epoch)


        logging.info("Training ended at epoch: {}".format(epoch))
        self.save_model("final")

    def log_losses(self, epoch):
        feed_dict = {self.model.x: self.val_xs,
                     self.model.s: self.val_s,
                     self.model.y: self.val_ys,
                     self.model.noise: self.val_noise}
        loggables = self.model.get_loggables()
        s = 'Stats on the validation set -- Epoch {:4} '.format(epoch)
        for name, tensor in loggables.items():
            loggable_value = self.session.run(tensor, feed_dict = feed_dict)
            s += '{}: {} '.format(name, loggable_value)
        logging.info(s)

    def log_stats_classifier(self, epoch, classifier=LogisticRegression):
        train_repr = self.session.run(self.model.model_last_hidden_layer, feed_dict = {
            self.model.x: self.train_xs, 
            self.model.noise: self.train_noise})
        val_repr = self.session.run(self.model.model_last_hidden_layer, feed_dict = {
            self.model.x: self.val_xs, 
            self.model.noise: self.val_noise})
        
        cl = classifier(solver="sag", max_iter=1000)
        cl.fit(train_repr, np.argmax(self.train_ys, axis=1))
        y_val_pred = cl.predict(val_repr)
        cl = classifier(solver="sag", max_iter=1000)
        cl.fit(train_repr, np.argmax(self.train_s, axis=1))
        s_val_pred = cl.predict(val_repr)
        
        s_val_acc  = sum( np.equal(s_val_pred, np.argmax(self.val_s, axis=1) )) / float(len(s_val_pred))
        y_val_acc  = sum( np.equal(y_val_pred, np.argmax(self.val_ys, axis=1) )) / float(len(y_val_pred))

        logging.info('Epoch {:4} y acc {:2.3f} s acc: {:2.3f}'.format(int(epoch[0]), y_val_acc, s_val_acc))

        return y_val_acc, s_val_acc

    def updateTensorboardStats(self, epoch):
        stat_des = self.session.run(self.model.train_stats, feed_dict = { 
            self.model.x:self.train_xs, 
            self.model.y:self.train_ys, 
            self.model.s: self.train_s,
            self.model.noise: self.train_noise })

        self.writer.add_summary(stat_des, global_step = epoch)

        stat_des = self.session.run(self.model.val_stats, feed_dict = { 
            self.model.x:self.val_xs, 
            self.model.y:self.val_ys, 
            self.model.s: self.val_s,
            self.model.noise: self.val_noise })
        self.writer.add_summary(stat_des, global_step = epoch)


    def save_model(self, epoch):
        if epoch == "final" or self.options.save_at_epoch(epoch):
            logging.info("Saving model (epoch:{})".format(epoch))
            self.saver.save(self.session, self.options.output_fname(epoch))
