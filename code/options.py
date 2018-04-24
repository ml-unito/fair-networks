import sys
import bank_marketing_dataset as ds
import tensorflow as tf

class Options:

    def __init__(self):
        self.dataset_name = "bank"
        self.dataset = ds.BankMarketingDataset()

        # self.num_features = 92   # Adult
        self.num_features = 51     # Bank

        self.epoch_start = 0
        self.epoch_end = 10000
        self.epochs = range(self.epoch_start, self.epoch_end)
        self.resume_learning = False

        self.epochs_per_save = 1000

    def parse_epochs(self, epochs_str):
        start, end = epochs_str.split(':')

        if start != '':
            self.epoch_start = int(start)
            self.resume_learning = True

        self.epoch_end = int(end)
        self.epochs = range(self.epoch_start, self.epoch_end)

    def parse(self, argv):
        if len(argv) < 2:
            print("Usage: %s <num_hidden_units> [epoch_start:epoch_end]" % argv[0])
            sys.exit(1)

        self.hidden_units = int(argv[1])

        if len(argv) == 3:
            self.parse_epochs(argv[2])

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
