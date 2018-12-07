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


sys.path.append('code')

from model import Model
from options import Options
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

def eval_accuracies_on_representation(file):
    df = pandas.read_csv(file)
    y_columns = df.filter(regex=("y.*")).values
    s_columns = df.filter(regex=("s.*")).values
    h_columns = df.filter(regex=("h.*")).values

    h_train, h_test, s_train, s_test, y_train, y_test = ms.train_test_split(h_columns, s_columns, y_columns, random_state=RANDOM_SEED)

    y_train = np.argmax(y_train, axis=1)
    y_test = np.argmax(y_test, axis=1)
    s_train = np.argmax(s_train, axis=1)
    s_test = np.argmax(s_test, axis=1)

    results = []
    results.append(svc_performances(h_train, y_train, h_test, y_test, s_train, s_test))
    results.append(trees_performances(
        h_train, y_train, h_test, y_test, s_train, s_test))
    results.append(logistic_regression_performances(
        h_train, y_train, h_test, y_test, s_train, s_test))
    return results


def logistic_regression_performances(h_train, y_train, h_test, y_test, s_train, s_test):
    logistic_regression_y = lm.LogisticRegression(random_state=RANDOM_SEED)

    logistic_regression_y.fit(h_train, y_train)
    pred_y_train = logistic_regression_y.predict(h_train)
    pred_y_test = logistic_regression_y.predict(h_test)

    acc_y_train = sum(pred_y_train == y_train)/float(len(y_train))
    acc_y_test = sum(pred_y_test == y_test)/float(len(y_test))

    logistic_regression_s = lm.LogisticRegression(random_state=RANDOM_SEED)

    logistic_regression_s.fit(h_train, s_train)

    pred_s_train = logistic_regression_s.predict(h_train)
    pred_s_test = logistic_regression_s.predict(h_test)

    acc_s_train = sum(pred_s_train == s_train)/float(len(s_train))
    acc_s_test = sum(pred_s_test == s_test)/float(len(s_test))

    return {"classifier": "logistic_regression", "accuracies": {"s": {"train": acc_s_train, "test": acc_s_test},
                                                                "y": {"train": acc_y_train, "test": acc_y_test}}}


def svc_performances(h_train, y_train, h_test, y_test, s_train, s_test):
    svc_y = svm.SVC(random_state=RANDOM_SEED)

    svc_y.fit(h_train, y_train)
    pred_y_train = svc_y.predict(h_train)
    pred_y_test = svc_y.predict(h_test)

    acc_y_train = sum(pred_y_train == y_train)/float(len(y_train))
    acc_y_test = sum(pred_y_test == y_test)/float(len(y_test))

    svc_s = svm.SVC(random_state=RANDOM_SEED)

    svc_s.fit(h_train, s_train)

    pred_s_train = svc_s.predict(h_train)
    pred_s_test = svc_s.predict(h_test)

    acc_s_train = sum(pred_s_train == s_train)/float(len(s_train))
    acc_s_test = sum(pred_s_test == s_test)/float(len(s_test))

    return { "classifier": "svc", "accuracies": { "s": {"train": acc_s_train, "test": acc_s_test},
                                                  "y": {"train": acc_y_train, "test": acc_y_test } } }


def trees_performances(h_train, y_train, h_test, y_test, s_train, s_test):
    tree_y = tree.DecisionTreeClassifier(max_depth=4, random_state=RANDOM_SEED)

    tree_y.fit(h_train, y_train)
    pred_y_train = tree_y.predict(h_train)
    pred_y_test = tree_y.predict(h_test)

    acc_y_train = sum(pred_y_train == y_train)/float(len(y_train))
    acc_y_test = sum(pred_y_test == y_test)/float(len(y_test))

    tree_s = tree.DecisionTreeClassifier(max_depth=4, random_state=RANDOM_SEED)

    tree_s.fit(h_train, s_train)

    pred_s_train = tree_s.predict(h_train)
    pred_s_test = tree_s.predict(h_test)

    acc_s_train = sum(pred_s_train == s_train)/float(len(s_train))
    acc_s_test = sum(pred_s_test == s_test)/float(len(s_test))

    return {"classifier": "tree", "accuracies": {"s": {"train": acc_s_train, "test": acc_s_test},
                                                "y": {"train": acc_y_train, "test": acc_y_test}}}

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


    experiments_results = []
    for file in os.listdir(representations_dir):
        results = eval_accuracies_on_representation(os.path.join(representations_dir, file))
        experiments_results.append( { "model_name":file, "results": results } )

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
