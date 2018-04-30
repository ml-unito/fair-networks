import sys
from bank_marketing_dataset import BankMarketingDataset
from adult_dataset import AdultDataset
import tensorflow as tf

class Options:
    def __init__(self):
        self.num_features = 108   # Adult
        # self.num_features = 51     # Bank

        self.epoch_start = 0
        self.epoch_end = 10000
        self.epochs = range(self.epoch_start, self.epoch_end)
        self.resume_learning = False

        self.epochs_per_save = 1000

    def parse_epochs(self, epochs_str):
        epochs_spec = epochs_str.split(':')
        if len(epochs_spec) == 1:
            self.epoch_end = int(epochs_spec[0])
            return

        start,end = epochs_spec

        if start != '':
            self.epoch_start = int(start)
            self.resume_learning = True

        if end != '':
            self.epoch_end = int(end)

        self.epochs = range(self.epoch_start, self.epoch_end)

    def print_usage(self):
        print("Usage: %s <dataset> <num_hidden_units> [epochs_specs]" % sys.argv[0])
        print("  dataset: choose from adult or bank")
        print("  epoch_specs specifies the range of epochs to work with;")
        print("  syntax is:  <start>:<end>")
        print("     with <start> defaulting to 0 and <end> defaulting to 10000")
        print("     giving a single number and omitting the colon will be ")
        print("     interpreted as :<end>")
        print("  examples:")
        print("     100:5000  -- epochs from 100 to 5000")
        print("     :5000     -- epochs from 0 to 5000")
        print("     5000      -- epochs from 0 to 5000")
        print("     100:      -- epochs from 100 to 10000")
        print("  NOTE: at test time <end> need to be set to the epoch of the")
        print("        model to be retrieved.")

    def parse_dataset(self, str):
        datasets = { 'adult': AdultDataset, 'bank': BankMarketingDataset }
        if str not in ['adult', 'bank']:
            self.print_usage()
            sys.exit(1)

        self.dataset_name = str
        self.dataset = datasets[str]()
        self.num_features = self.dataset.num_features()

    def parse(self, argv):
        if len(argv) < 3:
            self.print_usage()
            sys.exit(1)

        self.parse_dataset( argv[1] )

        self.hidden_units = int(argv[2])

        if len(argv) == 4:
            self.parse_epochs(argv[3])

        self.hidden_layers = [
            (self.hidden_units, tf.nn.sigmoid, tf.truncated_normal_initializer) # first layer
            ]

        self.exp_name = "%s_h%d" % (self.dataset_name, self.hidden_units)

        return self

    def model_fname(self, epoch):
        return "models/%s-epoch-%d.ckpt" % (self.exp_name, epoch)

    def log_fname(self):
        return 'logdir/log_%s' % self.exp_name

    def save_at_epoch(self, epoch):
        early_saves = epoch < 1000 and epoch % 100 == 0
        normal_saves = epoch % self.epochs_per_save == 0

        return early_saves or normal_saves
