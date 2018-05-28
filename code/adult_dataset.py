import csv
import requests
import os.path
import tensorflow as tf
import numpy as np

from tqdm import tqdm

from dataset_base import DatasetBase

class AdultDataset(DatasetBase):
    """
    Helper class allowing to download and load into memory the adult dataset
    """
    def all_columns(self):
        return [ "age","workclass","fnlwgt","education","education-num","marital-status","occupation","relationship","race","sex","capital-gain","capital-loss","hours-per-week","native-country","label" ]

    def one_hot_columns(self):
        return [ "workclass","education","marital-status","occupation","relationship","race","sex","native-country","label" ]

    def dataset_path(self):
        return "data/adult-train-test.csv"

    def sep(self):
        return ','

    def files_to_retrieve(self):
        return [
            ('https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data',
             'data/adult.data'),
            ('https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test',
             'data/adult.test')
        ]

    def sensible_columns(self):
        return["sex"]

    def y_columns(self):
        return ["label"]

    def prepare_all(self):
        with open(self.dataset_path(), "w") as file:
            file.write(','.join(self.all_columns())+"\n")

            with open("data/adult.data", "r") as train:
                for line in train:
                    file.write(line)

            with open("data/adult.test", "r") as test:
                for line in test:
                    if len(line) > 0 and line[0] == '|':
                        continue

                    file.write(line[:-2]+"\n")
