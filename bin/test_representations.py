#!/usr/bin/python

import pandas
import sklearn.model_selection as ms
import sklearn.svm as svm
import sklearn.tree as tree
import sklearn.linear_model as lm
import numpy as np
import os
import sys
import json
import traceback
import re
from termcolor import colored

RANDOM_SEED=42

from fair.fn.model import Model
from fair.utils.options import Options
from sklearn.metrics import confusion_matrix

# Reads representations from experiments in the given directory (must be the root of the 
# experiments directory, the one containg the config.json file) and writes a "perfomrances.json"
# file containing the performances of several learning algorithms on the representations built
# for the experiment.

def read_commit_id(file_path):
    if os.path.exists(file_path):
        return open(file_path, "r").read()
    else:
        return "N/A"

def accuracy(pred, y):
    return np.average(y == pred)

def eval_accuracies_on_representation(file):
    print("processing: {}".format(file))
    df = pandas.read_csv(file)
    y_columns = df.filter(regex=("y.*")).values
    s_columns = df.filter(regex=("s.*")).values
    h_columns = df.filter(regex=("h.*")).values

    h_train, h_test, s_train, s_test, y_train, y_test = ms.train_test_split(h_columns, s_columns, y_columns, random_state=RANDOM_SEED)
    

    y_train = np.argmax(y_train, axis=1)
    y_test = np.argmax(y_test, axis=1)
    s_train = np.argmax(s_train, axis=1)
    s_test = np.argmax(s_test, axis=1)

    datasets = (h_train, y_train, h_test, y_test, s_train, s_test)

    results = {}
    results["svc"] = test_performances(klass=svm.SVC, kargs={"random_state": RANDOM_SEED, "kernel":"linear"}, datasets=datasets)
    results["tree"] = test_performances(klass=tree.DecisionTreeClassifier, kargs={"random_state":RANDOM_SEED, "max_depth":4}, datasets=datasets )
    results["lr"] = test_performances(klass=lm.LogisticRegression, kargs={"random_state": RANDOM_SEED}, datasets=datasets)
    return results


def test_performances(klass, kargs, datasets):
    h_train, y_train, h_test, y_test, s_train, s_test = datasets
    classifier_y = klass(**kargs)

    classifier_y.fit(h_train, y_train)
    pred_y_train = classifier_y.predict(h_train)
    pred_y_test = classifier_y.predict(h_test)

    acc_y_train = accuracy(pred_y_train, y_train)
    acc_y_test = accuracy(pred_y_test, y_test)

    classifier_s = klass(**kargs)

    classifier_s.fit(h_train, s_train)

    pred_s_train = classifier_s.predict(h_train)
    pred_s_test = classifier_s.predict(h_test)

    acc_s_train = accuracy(pred_s_train, s_train)
    acc_s_test = accuracy(pred_s_test, s_test)

    return  {
        "s": {"train": acc_s_train, "test": acc_s_test},
        "y": {"train": acc_y_train, "test": acc_y_test}
        }


def process_dir(path):
    config_path = os.path.join(path, "config.json")
    if not os.path.exists(config_path):
        print(colored("Cannot find config file in %s -- Skipping to next directory" % path, "red"))
        return { "experiment_name": path, "error": "Cannot find config file" }


    opts = Options([None, config_path])
    representations_dir = os.path.join(path, "representations")

    if not os.path.exists(representations_dir):
        print("Directory %s does not exists." % representations_dir )
        print(colored("Skipping this directory", "red"))
        return { "experiment_name": path, "error": "Cannot find representation directory" }


    experiments_results = {}
    for file in os.listdir(representations_dir):
        results = eval_accuracies_on_representation(os.path.join(representations_dir, file))
        experiments_results[file] = results

    if len(experiments_results) == 0:
        return { "experiment_name": path, "error": "No representation found in dir %s" % representations_dir }

    return {
        "experiment_name": os.path.abspath(path),
        "commit_id": read_commit_id(os.path.join(path, "commit-id")),
        "config": opts.config_struct(),
        "performances": experiments_results
    }


# Read files in a given representations directory and write a report about each representation
# found there.
#
# output is a json file in the format:
#
# {
#  "experiment_name": ...
#  "config": {
#     ...
#  }
#  models_performances: [
#     { model_name: ..., acc_s: ..., acc_y: ... }
#     ...
#  ]
# }

# file = "experiments/adult/adult-fair-networks-representations.csv"
#
#file = sys.argv[1]

results = None
dir = sys.argv[1]

try:
    if re.search(r'^_.*', dir):
        print(colored("Name of directory %s starts with _, skipping to the next" % dir, "green"))
    else:
        experiment_base_dir = dir

        print(colored("Processing directory: %s" % experiment_base_dir, "green"))
        results = process_dir(experiment_base_dir)
except:
    error_info = sys.exc_info()
    results = {
        "experiment_name": os.path.abspath(dir),
        "error": str(error_info[1]),
        "trace:": traceback.format_exc()
    }

output_path = os.path.join(dir, "performances.json")
with open(output_path, "w") as file:
    file.write(json.dumps(results, indent=4))
