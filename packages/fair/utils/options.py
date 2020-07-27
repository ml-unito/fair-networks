import sys
from fair.datasets.bank_marketing_dataset import BankMarketingDataset
from fair.datasets.adult_dataset import AdultDataset
from fair.datasets.synth_dataset import SynthDataset
from fair.datasets.german_dataset import GermanDataset
from fair.datasets.german_louizos_dataset import GermanLouizosDataset
from fair.datasets.synth_easy_dataset import SynthEasyDataset
from fair.datasets.synth_easy2_dataset import SynthEasy2Dataset
from fair.datasets.synth_easy3_dataset import SynthEasy3Dataset
from fair.datasets.synth_easy4_dataset import SynthEasy4Dataset
from fair.datasets.yale_b_dataset import YaleBDataset
from fair.datasets.compas_dataset import CompasDataset
from fair.datasets.default_dataset import DefaultDataset
from fair.datasets.fake_news_dataset import FakeNewsDataset
import argparse
import textwrap
import os
import json
import re
import tensorflow.compat.v1 as tf
from copy import copy, deepcopy
from termcolor import colored
from pathlib import Path
import glob
import logging

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
    LAYER -> [slehni]?INT
    LAYER -> [slehni]?INT:LAYER

where the integers are the number of hidden units in the layer being specified and the
optional flags specify the activation unit to be used (s==sigmoid, l==linear, e==leaky_relu, h==tanh, n==noise, i==identity, default=='s').

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
    """Handles the options given to the main script.

        NOTE: all relative paths are assumed to be rooted in the same directory as the config file
           (defaults to the current directory when the config file is not provided)

        See the definition of the PARAMS_DESCRIPTION constant for a description of some of
        the options.

        One important facet of this handling is that options will loaded from json files
        when possible. Command line options can be used to overwrite values read from disk
        or for using the program without a json option file.

        self._try_load_opts and self._try_update_opts are the methods used to try loading the
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
        self.random_units: array of tuples specifying how many random neurons should be 
            contained in each hidden layer
        self.noise_type: string specifying which kind of activation to be used after the 
            noise layer. Defaults to 'default', which uses no activation.

        if result.schedule: array containing the schedule for the training of the network

        self.eval_stats: true if the task is simply to evaluate stats and exit
        self.eval_data_path: string representing the path where to store the representations built
            by the current model (if != None no training is performed)
        self.fairness_importance: numeric value representing how important the fairness constraint is

        self.epochs_per_save: number of epochs to be performed before saving a new model. This is
            used only epochs > 1000. Before this treshold a model is saved every 10 epochs.
    """

    HIDDEN_LAYER_SPEC_REGEXP = r'^([wnslrieh])?(\d+)?$'

    INITIALIZERS = {
        'constant': tf.initializers.constant,
        'glorot_normal': tf.initializers.glorot_uniform,
        'glorot_uniform': tf.initializers.glorot_uniform,
        'identity': tf.initializers.identity,
        'ones': tf.initializers.ones,
        'orthogonal': tf.initializers.orthogonal,
        'random_normal': tf.initializers.random_normal,
        'random_uniform': tf.initializers.random_uniform,
        'truncated_normal': tf.initializers.truncated_normal,
        'uniform_unit_scaling': tf.initializers.uniform_unit_scaling,
        'variance_scaling': tf.initializers.variance_scaling,
        'zeros': tf.initializers.zeros
    }

    DATASETS = {'adult': AdultDataset, 'bank': BankMarketingDataset,
                'german': GermanDataset, 'german-louizos': GermanLouizosDataset, 'synth': SynthDataset,
                'synth-easy': SynthEasyDataset, 'synth-easy2': SynthEasy2Dataset, 'synth-easy3': SynthEasy3Dataset,
                'yale': YaleBDataset, 'synth-easy4': SynthEasy4Dataset, 'compas': CompasDataset,
                'default': DefaultDataset, 'fakenews': FakeNewsDataset}


    def __init__(self, args):
        self.resume_learning = False
        self.epochs_per_save = 1000
        self.args = args

        self.used_options = self.parse(self.args)
        

        # with open(self.output_fname() + "_used_options.json", "w") as json_file:
        #     json_file.write(self.json_representation())

    def parse(self, argv):
        config_opts = self._try_load_opts(argv)

        parser = argparse.ArgumentParser(description=PARAMS_DESCRIPTION, formatter_class=argparse.RawDescriptionHelpFormatter)
        self._configure_parser(parser, checkpoint_already_given='checkpoint' in config_opts,
                               dataset_already_given='dataset' in config_opts)
        # it is important to retrieve this value before _try_update_opts. dirty workaround, sorry
        self.is_ddc = config_opts['ddc']
        result = self._try_update_opts(config_opts, parser.parse_args(argv[1:]))

        self._set_logging(result)
        self._set_initializers(result)
        self._set_epoch_save_ranges(result)
        self._set_datasets(result)
        self._set_layers(result)
        self._set_noise_type(result)
        self.fairness_importance = result.fairness_importance

        self.model_dir = result.model_dir
        self.resume_ckpt = result.resume_ckpt
        self.resume_learning = self.input_fname() != None

        self.batch_size = result.batch_size
        self.learning_rate = result.learning_rate

        if result.schedule != None:
            self.schedule = Schedule(result.schedule)

        self.eval_stats = result.eval_stats
        self.eval_data_path = self.path_for(result.eval_data)
        self.random_seed = result.random_seed

        self.var_loss = result.var_loss
        self.get_info = None if result.get_info == 'none' else result.get_info

        return result

    def print_config(self):
        logging.info(json.dumps(vars(self.used_options), indent=4))


    def path_for(self, path):
        if path == None:
            return None

        if os.path.isabs(path):
            return path

        return os.path.join(self.config_base_path, path)

    def _set_epoch_save_ranges(self, result):
        range_specs = getattr(result, "save_model_schedule", "2000:100-10000:500")
        logging.debug("_set_epoch_save - range_spec: {}".format(range_specs))
        start = 1
        step = None
        self._epoch_save_ranges = []
        for range_str in range_specs.split("-"):
            logging.debug("Splitting range spec: {}".format(range_str))
            end, step = [int(elem) for elem in range_str.split(":")]
            self._epoch_save_ranges.append((range(start, end), step))
            start = end
        self._epoch_save_ranges.append((range(start, sys.maxsize), step))

        logging.info("Saving at epochs: {}".format(self._epoch_save_ranges))


    def model_fname(self, epoch):
        return self.path_for("{}/model-{}.ckpt".format(self.model_dir, epoch))

    def output_fname(self, epoch):
        return self.model_fname(epoch)

    def input_fname(self):
        if self.resume_ckpt:
            return self.resume_ckpt

        return tf.train.latest_checkpoint(self.path_for(self.model_dir))

    def log_fname(self):
        return self.path_for('logdir/logs')

    def save_at_epoch(self, epoch):
        spec = [s for s in self._epoch_save_ranges if epoch in s[0]]

        if(len(spec) != 1):
            logging.error( "Epoch {} not in any of the allowed ranges (or in more than one). Exiting.".format(epoch))
            sys.exit(1)

        _, step = spec[0]
        return epoch % step == 0

    def config_struct(self):
        return vars(self.used_options)

    #  PRIVATE METHODS

    def _configure_parser(self, parser, checkpoint_already_given=None, dataset_already_given=None):
        parser.add_argument('-c', '--model-dir', type=str,
                            help="Name of the directory where to save the model.")
        parser.add_argument('--save-model-schedule', type=str, help="Schedule for saving models. Use format: E1:N1-E2:N2...:En:Nn"
                            ", where Ei is the i-th end-point and Ni is the frequency in the given range. "
                            "For instance: 100:10-1000:100-5000:200 says to save models every 10 epochs from epochs in range"
                            " 1:100 every 100 epochs for epochs in range 101:1000, and every 200 epochs for epochs in 1000:5000")                            
        parser.add_argument('-i', '--resume-ckpt', type=str,
                            help="Resume operations from the given ckpt, resume latest ckpt if not provided.")
        parser.add_argument('-H', '--hidden-layers',
                            type=str, help='hidden layers specs')
        parser.add_argument('-S', '--sensible-layers',
                            type=str, help='sensible network specs')
        parser.add_argument('-Y', '--class-layers', type=str,
                            help='output network specs')
        parser.add_argument('-r', '--random-seed', type=int,
                            help='sets the random seed used in the experiment')
        parser.add_argument('-e', '--eval-stats', default=False, action='store_const', const=True,
                            help='Evaluate all stats and print the result on the console (if set training options will be ignored)')
        parser.add_argument('-E', '--eval-data', metavar="PATH", type=str, 
                            help='Evaluate the current model on the whole dataset and save it to disk. Specifically a line (N(x),s,y) is saved for each example (x,s,y), where N(x) is the value computed on the last layer of "model" network.')
        parser.add_argument('-s', '--schedule', type=str,
                            help="Specifies how to schedule training epochs (see the main description for more information.)")
        parser.add_argument('-f', '--fairness-importance', type=float,
                            help="Specify how important is fairness w.r.t. the error")
        parser.add_argument('-d', '--dataset-base-path', type=str,
                            help="Specify the base directory for storing and reading datasets")
        parser.add_argument('-b', '--batch-size', type=int,
                            help="Specifies the batch size to be used")
        parser.add_argument('-l', '--learning-rate', type=float,
                            help="Specifies the (initial) learning rate")
        parser.add_argument('-g', '--get-info', choices=['epoch', 'variables', 'data-sample', 'out-sample',
                                                         'none'], default='none', help="Returns a textual representation of model parameters")
        parser.add_argument('-V', '--var-loss', action='store_const', const=True, default=False,
                            help="Use the s_loss variance (instead of the mean) to train the common layers.")
        parser.add_argument('-v', '--verbose', type=bool, default=False,
                            help="Print additional information onto the console (it is equivalent to --log-level=DEBUG)")
        parser.add_argument('--kernel-initializer', choices=list(self.INITIALIZERS.keys()),
                            help="Sets the initializer for the kernel term, defaults to glorot_uniform if not given or set to 'default'")
        parser.add_argument('--bias-initializer', choices=list(self.INITIALIZERS.keys()),
                            help="Sets the initializer function for the bias term, defaults to glorot_uniform if not given or set to 'default'")
        parser.add_argument('--noise-type', choices=["default", "sigmoid_full", "sigmoid_sep", "sigmoid_sep_2"], 
                            help="Choose the type of activation used after the noise layer.")
        parser.add_argument('--log-level', choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO")
        parser.add_argument('--log-file', type=str, help="Specifies the file to save logs, defaults to stdout")
        parser.add_argument('--ddc', action='store_const', default=False, const=True, help='Employ the deep domain confusion model.')

        if not dataset_already_given:
            parser.add_argument('dataset', choices=[
                                'adult', 'bank', 'german', 'german-louizos', 'synth', 'synth-easy', 'synth-easy2', 'synth-easy3', 'compas', 'default'], 
                                help="dataset to be loaded")

    def _parse_hidden_units(self, spec):
        match = re.search(self.HIDDEN_LAYER_SPEC_REGEXP, spec)
        if match == None:
            raise ParseError(
                'Cannot parse layer specification for element:' + spec)

        if match.group(1) == 'n':
            return None

        if match.group(1) == 'i':
            return None

        if match.group(2) == None:
            raise ParseError(
                'Cannot parse layer specification for element {}: number of units missing from the specification'.format(spec))

        return int(match.group(2))

    def _parse_activation(self, spec):
        match = re.search(self.HIDDEN_LAYER_SPEC_REGEXP, spec)
        if match == None:
            raise ParseError(
                'Cannot parse layer specification for element:' + spec)

        if match.group(1) == None or match.group(1) == 's':
            return tf.nn.sigmoid

        if match.group(1) == 'l':
            return None

        if match.group(1) == 'r':
            return tf.nn.relu

        if match.group(1) == 'e':
            return tf.nn.leaky_relu

        if match.group(1) == 'h':
            return tf.nn.tanh

        if match.group(1) == 'i':
            return tf.identity

        if match.group(1) == 'n':
            return None

        if match.group(1) == 'w':
            # if we want a whiteout layer, it is going to have sigmoids. this is not ideal,
            # but sigmoids are all we are using in the paper-reported experiments.
            return tf.nn.sigmoid 

        raise ParseError(
            'Error in parsing layer specification for element:' + spec + '. This is a bug.')

    def _fix_num_layers_for_noise_layers(self, layers):
        """
        Noise layers are "particular" since they do not provide the number of hidden units they work on.
        They implicitly use the same number of features as the previous layer. This method assumes that
        the parsing methods have set that information to None and update it by copying the number of
        features of the previous layer (of the input dataset if no previous layer exists).
        """
        result = [[None, self.dataset.num_features()]]
        for line in layers:
            result.append(list(line))
            if line[0] == 'n' or line[0] == 'i':
                result[-1][1] = result[-2][1]
        return result[1:]

    def _parse_layers(self, str):
        layers_specs = str.split(':')
        initializers = (
            self.INITIALIZERS[self.kernel_intializer],
            self.INITIALIZERS[self.bias_initializer])

        result = [(self._parse_layer_type(spec), self._parse_hidden_units(spec), self._parse_activation(spec), initializers)
                  for spec in layers_specs]

        return self._fix_num_layers_for_noise_layers(result)

    def _parse_layer_type(self, spec):
        if spec == 'n':
            return 'n'
        elif spec == 'i':
            return 'i'
        elif spec[0] == 'w':
            return 'w'
        else:
            return None

    def _check_layers_specs(self, from_json=False):
        if self.hidden_layers_specs != None and self.sensible_layers_specs != None and self.class_layers_specs != None:
            return

        if from_json:
            raise ParseError(
                'Cannot parse layer specs read from the json options.')
        else:
            print({"hidden_layers_specs": self.hidden_layers_specs, "sensible_layers_specs":
                   self.sensible_layers_specs, "class_layers_specs": self.class_layers_specs})
            raise ParseError(
                'Cannot parse layer specs from options on the command line.')

    def _set_layers(self, options):
        self.hidden_layers_specs = options.hidden_layers
        self.sensible_layers_specs = options.sensible_layers
        self.class_layers_specs = options.class_layers

        self._check_layers_specs(from_json=False)

        self.hidden_layers = self._parse_layers(self.hidden_layers_specs)
        self.sensible_layers = self._parse_layers(self.sensible_layers_specs)
        self.class_layers = self._parse_layers(self.class_layers_specs)

    def _try_load_opts(self, argv):
        if len(argv) >= 2 and Path(argv[1]).is_file():
            file_to_read = argv[1]
            argv.pop(1)
            self.config_base_path = os.path.dirname(file_to_read)
            return json.loads(open(file_to_read).read())
        else:
            self.config_base_path = os.getcwd()

        if Path('.fn-config').is_file():
            return json.loads(open('.fn-config').read())

        return {}

    def _try_update_opts(self, config_opts, parsed_args):
        setted_args = {k: v for k, v in vars(parsed_args).items() if v != None}
        config_opts.update(setted_args)
        parsed_args.__dict__ = config_opts
        return parsed_args


    def _set_logging(self, result):
        self.verbose = result.verbose

        if self.verbose:
            self.log_level = logging.DEBUG
        else:
            self.log_level = logging.getLevelName(result.log_level)

        log_file = getattr(result, "log_file", None)
        
        if log_file!=None:
            logging.basicConfig(filename=log_file, level=self.log_level)
        else:
            logging.basicConfig(level=self.log_level)

    def _set_initializers(self, result):
        self.kernel_intializer = getattr(result, 'kernel_initializer', 'glorot_uniform')
        self.bias_initializer = getattr(result, 'bias_initializer', 'glorot_uniform')

        if self.kernel_intializer == "" or self.kernel_intializer == "default":
            self.kernel_intializer = "glorot_uniform"

        if self.bias_initializer == "" or self.bias_initializer == "default":
            self.bias_initializer = "glorot_uniform"

        logging.debug('Using kernel initializer: {}'.format(self.kernel_intializer))
        logging.debug('Using bias initializer: {}'.format(self.bias_initializer))

    def _set_noise_type(self, result):
        self.noise_type = getattr(result, 'noise_type', 'default')

        logging.debug('Using noise type: {}'.format(self.noise_type))


    def _set_datasets(self, result):
        self.dataset_name = result.dataset
        self.dataset_base_path = self.path_for(result.dataset_base_path)

        self.dataset = self.DATASETS[self.dataset_name](self.dataset_base_path)
        self.num_features = self.dataset.num_features()

