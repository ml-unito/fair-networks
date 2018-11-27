import sys
from bank_marketing_dataset import BankMarketingDataset
from adult_dataset import AdultDataset
from synth_dataset import SynthDataset
from german_dataset import GermanDataset
import argparse
import textwrap
import os
import json
import re
from copy import copy, deepcopy
from termcolor import colored
from pathlib import Path

PARAMS_DESCRIPTION = """\
or:

 fair_networks.py CONFIG_FILENAME [other options]

or:

 fair_networks.py

If CONFIG_FILENAME is given, the options found therein are used as
initial option values. Any option given on the command line overwrite
the ones in the file.

If no option is given CONFIG_FILENAME is assumed to be the file '.fn-config'
in the current directory (if exists).

The SCHEDULE option specifies a training procedure by enumerating the number of epochs
that should be spent optimizing different parts of the network and how.
The syntax is:
    SCHEDULE -> mINT:cINT
    INT -> {1..9}{0..9}*

where 'm' and 'c' specify the network part to be trained:
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

class ParseError(Exception):
    """Exception raised for errors in the command line options.

    Attributes:
        message -- explanation of the error
    """

    def __init__(self, message):
        self.message = message

class Schedule:
    """Simplifies scheduling of tasks specified by the options

    Attributes:
        schedule_specs: string array containing the components of the schedule as specified
            in the options.
        num_epochs: number of epochs to be performed following this schedule
        sub_nets_num_it: numbe of epochs for training the s and y subworks
    """
    def __init__(self, schedule_str):
        self.schedule_specs = schedule_str.split(':')
        self.num_epochs = self.parse_schedule('m',self.schedule_specs[0])
        self.sub_nets_num_it = self.parse_schedule('c',self.schedule_specs[1])

    def parse_schedule(self, kind, epoch_spec):
        if epoch_spec[0] != kind:
            raise ParseError("Cannot parse schedule option '%s' expected '%s' found" % (kind, epoch_spec[0]))

        return int(epoch_spec[1:])

class Options:
    """Handles the options given to the main script

        See the definition of the PARAMS_DESCRIPTION constant for a description of some of
        the options.

        One important facet of this handling is that options will loaded from json files
        when possible. Command line options can be used to overwrite values read from disk
        or for using the program without a json option file.

        self.try_load_opts and self.try_update_opts are the methods used to try loading the
        options from file and to merge command line options with those read from file.

        # Attributes

        After initializations a property will be set for each option succesfully recognised:

        self.dataset_name: dataset name to be used
        self.dataset_base_path: directory where dataset are stored
        self.dataset: an object instantiated to a subclass of DatasetBase allowing
            working with the given dataset

        self.num_features: number of features for the input data (depends on the dataset chosen)
        self.checkpoint_output: path name where to store checkpoints
        self.resume_learning: boolean specifying if a checkpoint needs to be restored

        self.hidden_layers: array of tuples specifying how to build the hidden layer
        self.sensible_layers: array of tuples specifying how to build the sensible layer
        self.class_layers: array of tuples specifying how to build the class layer

        if result.schedule: array containing the schedule for the training of the network

        self.eval_stats: true if the task is simply to evaluate stats and exit
        self.eval_data_path: string representing the path where to store the representations built
            by the current model (if != None no training is performed)
        self.fairness_importance: numeric value representing how important the fairness constraint is

        self.epochs_per_save: number of epochs to be performed before saving a new model. This is
            used only epochs > 1000. Before this treshold a model is saved every 10 epochs.
    """
    def __init__(self):
        self.resume_learning = False
        self.epochs_per_save = 1000

        used_options = self.parse(sys.argv)

        with open(self.output_fname() + "_used_options.json", "w") as json_file:
            json_file.write(json.dumps(vars(used_options), sort_keys=True,indent=4))

        print("Initializing system using options:")
        print(json.dumps(vars(used_options), indent=4))

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
            print( { "hidden_layers_specs":self.hidden_layers_specs, "sensible_layers_specs":self.sensible_layers_specs, "class_layers_specs":self.class_layers_specs})
            print(colored('Cannot parse layer specs from options on the command line.', 'red'))
            exit(1)



    def set_layers(self, options):
        if self.resume_learning:
            with open(self.input_fname() + "_used_options.json", "r") as json_file:
                parsed_json = json.loads(json_file.read())
            self.hidden_layers_specs = parsed_json['hidden_layers']
            self.sensible_layers_specs = parsed_json['sensible_layers']
            self.class_layers_specs = parsed_json['class_layers']

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

    def try_load_opts(self, argv):
        if len(argv) >= 2 and Path(argv[1]).is_file():
            file_to_read = argv[1]
            argv.pop(1)
            return json.loads(open(file_to_read).read())

        if Path('.fn-config').is_file():
            return json.loads(open('.fn-config').read())

        return {}

    def try_update_opts(self, config_opts, parsed_args):
        setted_args = { k:v for k,v in vars(parsed_args).items() if v != None }
        config_opts.update(setted_args)
        parsed_args.__dict__ = config_opts
        return parsed_args

    def configure_parser(self, parser, checkpoint_already_given=None, dataset_already_given=None):
        parser.add_argument('-c', '--checkpoint', metavar="FILENAME.ckpt", required=not checkpoint_already_given, type=str, help="Name of the checkpoint to be saved/loaded.")
        parser.add_argument('-o', '--output', metavar="FILENAME.ckpt", type=str, help="Name of the checkpoint to be saved into. Defaults to the value given to the -c parameter if not given.")
        parser.add_argument('-H', '--hidden-layers', type=str, help='hidden layers specs')
        parser.add_argument('-S', '--sensible-layers', type=str, help='sensible network specs')
        parser.add_argument('-Y', '--class-layers', type=str, help='output network specs')
        parser.add_argument('-r', '--random-seed', type=int, help='sets the random seed used in the experiment')
        parser.add_argument('-e', '--eval-stats', default=False, action='store_const', const=True, help='Evaluate all stats and print the result on the console (if set training options will be ignored)')
        parser.add_argument('-E', '--eval-data', metavar="PATH", type=str, help='Evaluate the current model on the whole dataset and save it to disk. Specifically a line (N(x),s,y) is saved for each example (x,s,y), where N(x) is the value computed on the last layer of "model" network.')
        parser.add_argument('-s', '--schedule', type=str, help="Specifies how to schedule training epochs (see the main description for more information.)")
        parser.add_argument('-f', '--fairness-importance', type=float, help="Specify how important is fairness w.r.t. the error")
        parser.add_argument('-d', '--dataset-base-path', type=str, help="Specify the base directory for storing and reading datasets")

        if not dataset_already_given:
            parser.add_argument('dataset', choices=['adult', 'bank', 'german', 'synth'], help="dataset to be loaded")

    def parse(self, argv):
        config_opts = self.try_load_opts(argv)

        datasets = { 'adult': AdultDataset, 'bank': BankMarketingDataset, 'german':GermanDataset, 'synth': SynthDataset }
        parser = argparse.ArgumentParser(description=PARAMS_DESCRIPTION, formatter_class=argparse.RawDescriptionHelpFormatter)

        self.configure_parser(parser, checkpoint_already_given='checkpoint' in config_opts, dataset_already_given='dataset' in config_opts)

        result = self.try_update_opts(config_opts, parser.parse_args(argv[1:]))

        self.dataset_name = result.dataset
        self.dataset_base_path = result.dataset_base_path
        print("dataset base path:%s" % (self.dataset_base_path))

        self.dataset = datasets[self.dataset_name](self.dataset_base_path)
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
        self.fairness_importance = result.fairness_importance
        self.random_seed = result.random_seed

        return result

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
        early_saves = epoch < 1000 and epoch % 10 == 0
        normal_saves = epoch % self.epochs_per_save == 0

        return early_saves or normal_saves
