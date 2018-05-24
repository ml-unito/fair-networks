import sys
from bank_marketing_dataset import BankMarketingDataset
from adult_dataset import AdultDataset
from synth_dataset import SynthDataset
import tensorflow as tf
import argparse
import textwrap

class Options:
    def __init__(self):
        # self.num_features = 108   # Adult
        # self.num_features = 51     # Bank

        self.epoch_start = 0
        self.epoch_end = 10000
        self.epochs = range(self.epoch_start, self.epoch_end)
        self.resume_learning = False

        self.epochs_per_save = 1000

        self.parse(sys.argv)

        self.exp_name = "%s_h%s" % (self.dataset_name, self.hidden_layers_specs)



    def parse_epochs(self, epochs_str):
        epochs_spec = epochs_str.split(':')
        if len(epochs_spec) == 1:
            self.epoch_end = int(epochs_spec[0])
            self.epochs = range(self.epoch_start, self.epoch_end)

            return

        start,end = epochs_spec

        if start != '':
            self.epoch_start = int(start)
            self.resume_learning = True

        if end != '':
            self.epoch_end = int(end)

        self.epochs = range(self.epoch_start, self.epoch_end)

    def parse_layers(self, str):
        self.hidden_layers_specs = str
        layers_specs = str.split(':')
        self.hidden_layers = [
            (int(hidden_units), tf.nn.sigmoid, tf.truncated_normal_initializer)
               for hidden_units in layers_specs ]

    def parse(self, argv):
        description = """\
        epoch_specs specifies the range of epochs to work with;
        syntax is:  <start>:<end>

        with <start> defaulting to 0 and <end> defaulting to 10000
        giving a single number and omitting the colon will be
        interpreted as :<end>

        examples:
            100:5000  -- epochs from 100 to 5000
            :5000     -- epochs from 0 to 5000
            5000      -- epochs from 0 to 5000
            100:      -- epochs from 100 to 10000
        NOTE: at test time <end> need to be set to the epoch of the
            model to be retrieved.

        hidden_layers specifies the composition of each hidden layer;
        syntax is: h_1:h_2:...:h_K

        where h_i being the number of hidden units in the i-th layer.

        examples:
             10       -- a single hidden layer with 10 neurons
             10:5:2   -- three hidden layers with 10, 5, and 2 neuron respectively
                            """
        datasets = { 'adult': AdultDataset, 'bank': BankMarketingDataset, 'synth': SynthDataset }
        parser = argparse.ArgumentParser(description=description,formatter_class=argparse.RawDescriptionHelpFormatter)
        parser.add_argument('dataset', choices=['adult', 'bank', 'synth'], help="dataset to be loaded")
        parser.add_argument('hidden_layers', type=str, help='hidden layers specs')
        parser.add_argument('epoch_specs', help = 'which epochs to be run')
        result = parser.parse_args()


        self.dataset_name = result.dataset
        self.dataset = datasets[self.dataset_name]()
        self.num_features = self.dataset.num_features()

        self.num_features = self.dataset.num_features()

        self.parse_layers(result.hidden_layers)
        self.parse_epochs(result.epoch_specs)

        return self

    def model_fname(self, epoch):
        return "models/%s-epoch-%d.ckpt" % (self.exp_name, epoch)

    def log_fname(self):
        return 'logdir/log_%s' % self.exp_name

    def save_at_epoch(self, epoch):
        early_saves = epoch < 1000 and epoch % 100 == 0
        normal_saves = epoch % self.epochs_per_save == 0

        return early_saves or normal_saves
