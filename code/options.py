import sys
from bank_marketing_dataset import BankMarketingDataset
from adult_dataset import AdultDataset
from synth_dataset import SynthDataset
import tensorflow as tf
import argparse
import textwrap
import os
import json
import re
from copy import copy, deepcopy
from termcolor import colored

class ParseError(Exception):
    """Exception raised for errors in the command line options.

    Attributes:
        message -- explanation of the error
    """

    def __init__(self, message):
        self.message = message

class Schedule:
    def __init__(self, schedule_str):
        self.schedule_specs = schedule_str.split(':')
        self.num_epochs = self.parse_schedule('m',self.schedule_specs[0])
        self.sub_nets_num_it = self.parse_schedule('c',self.schedule_specs[1])

    def parse_schedule(self, kind, epoch_spec):
        if epoch_spec[0] != kind:
            raise ParseError("Cannot parse schedule option '%s' expected '%s' found" % (kind, epoch_spec[0]))

        return int(epoch_spec[1:])

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

        with open(self.output_fname() + ".json", "w") as json_file:
            json_file.write(json.dumps(tbs, sort_keys=True,indent=4))

    def parse_hidden_units(self, spec):
        match = re.search('^[sl]?(\d+)$', spec)
        if match == None:
            print(colored('Cannot parse layer specification for element:' + spec, 'red'))
            exit(1)

        return int(match.group(1))

    def parse_activation(self, spec):
        match = re.search('^([sl]?)\d+$', spec)
        if match == None:
            print(colored('Cannot parse layer specification for element:' + spec, 'red'))
            exit(1)


        if match.group(1) == '' or match.group(1) == 's':
            return tf.nn.sigmoid

        if match.group(1) == 'l':
            return None

        print(colored('Error in parsing layer specification for element:' + spec + '. This is a bug.', 'red'))
        exit(1)

    def parse_layers(self, str):
        layers_specs = str.split(':')
        return [(self.parse_hidden_units(spec), self.parse_activation(spec), tf.truncated_normal_initializer)
               for spec in layers_specs ]

    def check_layers_specs(self, from_json=False):
        if self.hidden_layers_specs != None and self.sensible_layers_specs != None and self.class_layers_specs != None:
            return

        if from_json:
            print(colored('Cannot parse layer specs read from the json options.', 'red'))
            exit(1)
        else:
            print(colored('Cannot parse layer specs from options on the command line.', 'red'))
            exit(1)



    def set_layers(self, options):
        if self.resume_learning:
            with open(self.input_fname() + ".json", "r") as json_file:
                parsed_json = json.loads(json_file.read())
            self.hidden_layers_specs = parsed_json['hidden_layers_specs']
            self.sensible_layers_specs = parsed_json['sensible_layers_specs']
            self.class_layers_specs = parsed_json['class_layers_specs']

            self.check_layers_specs(from_json=True)
        else:
            self.hidden_layers_specs = options.hidden_layers
            self.sensible_layers_specs = options.sensible_layers
            self.class_layers_specs = options.class_layers

            self.check_layers_specs(from_json=False)

        print(colored("Basing model on specs: H%s S%s Y%s" % (self.hidden_layers_specs, self.sensible_layers_specs, self.class_layers_specs), 'yellow'))

        self.hidden_layers = self.parse_layers(self.hidden_layers_specs)
        self.sensible_layers = self.parse_layers(self.sensible_layers_specs)
        self.class_layers = self.parse_layers(self.class_layers_specs)

    def parse(self, argv):
        description = """\
        The SCHEDULE option specifies a training procedure by enumerating the number of epochs
        that should be spent optimizing different parts of the network and how.
        The syntax is:
            SCHEDULE -> mINT:cINT
            INT -> {1..9}{0..9}*

        where 'a' and 's' specify the network part to be trained:
            - m [model], train the main model
            - c [classifiers], train the attached classifiers (y and s)

        the integer after the 'm' option specifies the total number of epochs to be performed;
        the integer after the 'c' option specifies how much the sensible and y layer have to be trained
        at each iteration.

        example:
            m100:c100        -- train the whole network for 10 epochs; for each batch train
                                the sensible and y network for 100 iterations before using it to
                                feedback the prediction to train model

        "*_LAYERS" options specify the composition of sub networks;
        syntax is:
            LAYER -> [sl]?INT
            LAYER -> [sl]?INT:LAYER

        where the integers are the number of hidden units in the layer being specified and the
        optional 's' or 'l' flags specify the activation unit to be used (s==sigmoid, l==linear,
        default=='s').

        examples:
             10       -- a single layer with 10 sigmoid neurons
             10:5:2   -- three layers with 10, 5, and 2 neuron respectively, allo sigmoid units
             l5:s3    -- two layers, first has 5 linear units, the second has 3 sigmoid units

        NOTE: layers options CAN be omitted when the model is restored from a .ckpt file
        """

        datasets = { 'adult': AdultDataset, 'bank': BankMarketingDataset, 'synth': SynthDataset }
        parser = argparse.ArgumentParser(description=description,formatter_class=argparse.RawDescriptionHelpFormatter)
        parser.add_argument('-c', '--checkpoint', metavar="FILENAME.ckpt", required=True, type=str, help="Name of the checkpoint to be saved/loaded.")
        parser.add_argument('-o', '--output', metavar="FILENAME.ckpt", type=str, help="Name of the checkpoint to be saved into. Defaults to the value given to the -c parameter if not given.")
        parser.add_argument('-H', '--hidden-layers', type=str, help='hidden layers specs')
        parser.add_argument('-S', '--sensible-layers', type=str, help='sensible network specs')
        parser.add_argument('-Y', '--class-layers', type=str, help='output network specs')
        parser.add_argument('-e', '--eval-stats', default=False, action='store_const', const=True, help='Evaluate all stats and print the result on the console (if set training options will be ignored)')
        parser.add_argument('-E', '--eval-data', metavar="PATH", type=str, help='Evaluate the current model on the whole dataset and save it to disk. Specifically a line (N(x),s,y) is saved for each example (x,s,y), where N(x) is the value computed on the last layer of "model" network.')
        parser.add_argument('-s', '--schedule', type=str, help="Specifies how to schedule training epochs (see the main description for more information.)")
        parser.add_argument('dataset', choices=['adult', 'bank', 'synth'], help="dataset to be loaded")
        result = parser.parse_args()
        self.dataset_name = result.dataset
        self.dataset = datasets[self.dataset_name]()
        self.num_features = self.dataset.num_features()


        if result.output == None:
            self.checkpoint_output = result.checkpoint
        else:
            self.checkpoint_output = result.output

        self._model_fname = result.checkpoint
        self.resume_learning = tf.train.checkpoint_exists(self.input_fname())

        self.set_layers(result)

        if result.schedule != None:
            self.schedule = Schedule(result.schedule)

        self.eval_stats = result.eval_stats
        self.eval_data_path = result.eval_data

        return self

    def model_fname(self, name):
        return "%s" % (name)

    def output_fname(self):
        return self.model_fname(self.checkpoint_output)

    def input_fname(self):
        return self.model_fname(self._model_fname)

    def log_fname(self):
        name = self.output_fname().split('/')[-1]
        name = name.split('.ckpt')[0]
        return 'logdir/log_%s' % name

    def save_at_epoch(self, epoch):
        early_saves = epoch < 100000 and epoch % 100 == 0
        normal_saves = epoch % self.epochs_per_save == 0

        return early_saves or normal_saves
