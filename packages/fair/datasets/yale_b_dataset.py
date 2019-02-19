import csv
import requests
import os.path
import tensorflow as tf
import numpy as np

from tqdm import tqdm

from .dataset_base import DatasetBase

class YaleBDataset(DatasetBase):
    """
    Helper class allowing to load into memory the Yale Faces B dataset
    """
    def __init__(self, workingdir):
        """
        Downloads and load into memory the dataset.
        """
        self.workingdir = workingdir
        self.load_all_separate_paths()

    def name(self):
        return "yale faces B"

    def all_columns(self):
        x_features = range(0, 504)
        cols = ['x_' + str(x) for x in x_features]
        cols.extend('y')
        cols.extend('s')
        return cols

    def one_hot_columns(self):
        return ['y', 's']

    def dataset_path(self):
        return "%s/yale_train.csv" % (self.workingdir)

    def sep(self):
        return ','

    def files_to_retrieve(self):
        pass

    def sensible_columns(self):
        return["s"]

    def y_columns(self):
        return ["y"]

    def prepare_all(self):
        pass

    def train_path(self):
        return "%s/yale_train.csv" % (self.workingdir)

    def test_path(self):
        return "%s/yale_test.csv"  % (self.workingdir)
        
