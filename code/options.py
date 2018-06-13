import sys
from bank_marketing_dataset import BankMarketingDataset
from adult_dataset import AdultDataset
from synth_dataset import SynthDataset
import tensorflow as tf
import argparse
import textwrap
import os
import json
from copy import copy

class Schedule:
    def __init__(self, schedule_list):
        self.schedule_list = schedule_list
        self.current_index = 0

    def get_next(self):
        try:
            current_part_tuple = self.schedule_list[self.current_index]
        except IndexError:
            return None
        if current_part_tuple[1] > 0:
            self.schedule_list[self.current_index] = (current_part_tuple[0], current_part_tuple[1] - 1)
            return current_part_tuple[0]
        else:
            self.current_index += 1
            return self.get_next()
        return None

    def is_s_next(self):
        current_part_tuple = self.schedule_list[self.current_index]
        try:
            next_part_tuple = self.schedule_list[self.current_index+1]
            return current_part_tuple[1] == 0 and next_part_tuple[0] == 's'
        except IndexError:
            return False

class Options:
    def __init__(self):
        self.resume_learning = False
        self.epochs_per_save = 1000

        self.parse(sys.argv)

        # self.exp_name = "%s_h%s_s%s_y%s" % (self.dataset_name, self.hidden_layers_specs, self.sensible_layers_specs, self.class_layers_specs)



        tbs = copy(self.__dict__)
        tbs.pop('dataset')
        for key in tbs.keys():
            tbs[key] = str(tbs[key])

        # tbs['hidden_layers'] = str(to_be_serialized['hidden_layers'])
        # tbs['sensible_layers'] = str(to_be_serialized['hidden_layers'])
        # tbs['class_layers'] =
        # print(to_be_serialized)

        with open(self.model_fname() + ".json", "w") as json_file:
            json_file.write(json.dumps(tbs, sort_keys=True,indent=4))

    def parse_layers(self, str):
        layers_specs = str.split(':')
        return [(int(hidden_units), tf.nn.sigmoid, tf.truncated_normal_initializer)
               for hidden_units in layers_specs ]


    def parse_schedule(self, str):
        schedule_specs = str.split(':')
        schedule_list = [(spec[0], int(spec[1:])) for spec in schedule_specs]
        schedule = Schedule(schedule_list)
        return schedule

    def parse(self, argv):
        description = """\
        The SCHEDULE option specifies a training procedure by enumerating the number of epochs
        that should be spent optimizing different parts of the network and how.
        The syntax is:
            SCHEDULE -> SCHEDULE_SPEC
            SCHEDULE -> SCHEDULE_SPEC:SCHEDULE
            SCHEDULE_SPECH -> aINT | sINT | yINT | hINT | xINT
            INT -> {1..9}{0..9}*

        where a,s,y,h and x specify the network part to be trained:
            - a [all],
            - s [sensible],
            - y [target],
            - h [hidden],
            - x [un-train sensible]

        examples:
            a10:s100:y100   -- train the whole network for 10 epochs; then the
                               section predicting s for 100 epochs; then the
                               section predicting y for 100 epochs.
            s10:x10         -- train the section predicting s for 10 epochs, then
                               un-train it for 10 epochs.

        "*_LAYERS" options specify the composition of sub networks;
        syntax is:
            LAYER -> INT
            LAYER -> INT:LAYER

        where the integers are the number of hidden units in the layer being specified

        examples:
             10       -- a single layer with 10 neurons
             10:5:2   -- three layers with 10, 5, and 2 neuron respectively
        """

        datasets = { 'adult': AdultDataset, 'bank': BankMarketingDataset, 'synth': SynthDataset }
        parser = argparse.ArgumentParser(description=description,formatter_class=argparse.RawDescriptionHelpFormatter)
        parser.add_argument('-c', '--checkpoint', metavar="FILENAME.ckpt", required=True, type=str, help="Name of the checkpoint to be saved/loaded.")
        parser.add_argument('-H', '--hidden-layers', type=str, help='hidden layers specs', required=True)
        parser.add_argument('-S', '--sensible-layers', type=str, help='sensible network specs', required=True)
        parser.add_argument('-Y', '--class-layers', type=str, help='output network specs', required=True)
        parser.add_argument('-e', '--eval-stats', default=False, action='store_const', const=True, help='Evaluate all stats and print the result on the console (if set training options will be ignored)')
        parser.add_argument('-s', '--schedule', type=str, help="Specifies how to schedule training epochs (see the main description for more information.)")
        parser.add_argument('dataset', choices=['adult', 'bank', 'synth'], help="dataset to be loaded")
        result = parser.parse_args()

        self.dataset_name = result.dataset
        self.dataset = datasets[self.dataset_name]()
        self.num_features = self.dataset.num_features()

        self.hidden_layers_specs = result.hidden_layers
        self.hidden_layers = self.parse_layers(result.hidden_layers)

        self.sensible_layers_specs = result.sensible_layers
        self.sensible_layers = self.parse_layers(result.sensible_layers)

        self.class_layers_specs = result.class_layers
        self.class_layers = self.parse_layers(result.class_layers)

        self._model_fname = result.checkpoint
        self.resume_learning = os.path.isfile(self.model_fname())

        print(self.hidden_layers)
        print(self.sensible_layers)
        print(self.class_layers)

        if result.schedule != None:
            self.schedule = self.parse_schedule(result.schedule)

        self.eval_stats = result.eval_stats

        return self

    def model_fname(self):
        return "%s" % (self._model_fname)

    def log_fname(self):
        name = self._model_fname.split('/')[-1]
        name = name.split('.ckpt')[0]
        return 'logdir/log_%s' % name

    def save_at_epoch(self, epoch):
        early_saves = epoch < 100000 and epoch % 100 == 0
        normal_saves = epoch % self.epochs_per_save == 0

        return early_saves or normal_saves
