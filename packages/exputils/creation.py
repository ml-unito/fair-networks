import os
import sys
from shutil import copy
from termcolor import colored

def check_dir_structure(target_dir):
    if not os.path.exists('experiments'):
        print(colored('Cannot find experiments/ dir under active dir.', "red"))
        sys.exit(1)
    if os.path.exists('experiments/{}'.format(target_dir)):
        print(colored('Experiment folder experiments/{} exists already.'.format(target_dir), "red"))
        sys.exit(1)

def create_experiment_dir(source_json, experiment_name):
    if 'experiments/' in experiment_name:
        experiment_name = experiment_name.replace('experiments/', '')
    os.makedirs('experiments/{}/representations'.format(experiment_name))
    os.makedirs('experiments/{}/models'.format(experiment_name))
    os.makedirs('experiments/{}/logdir'.format(experiment_name))
    copy(source_json, 'experiments/{}/'.format(experiment_name))

def check_and_create(source_json, experiment_name):
    check_dir_structure(experiment_name)
    create_experiment_dir(source_json, experiment_name)