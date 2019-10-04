import csv
import requests
import os.path
import tensorflow.compat.v1 as tf
import numpy as np
import zipfile
import pandas
import sklearn
import logging

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from fair.datasets.dataset_base import DatasetBase

from tqdm import tqdm


class DefaultDataset(DatasetBase):
    DOWNLOAD_PATH = 'data/default.csv'

    """
    Helper class allowing to download and load into memory the adult dataset
    """

    def name(self):
        return "Default"

    def all_columns(self):
        return ["limit_bal",  "sex",  "education",  "marriage",  "age",  "pay_0",  "pay_2",  "pay_3", 
                "pay_4",  "pay_5",  "pay_6",  "bill_amt1",  "bill_amt2",  "bill_amt3",  "bill_amt4", 
                "bill_amt5",  "bill_amt6",  "pay_amt1",  "pay_amt2",  "pay_amt3",  "pay_amt4",  "pay_amt5",  
                "pay_amt6",  "y"]

    def one_hot_columns(self):
        return ["sex", "education", "marriage", "y"]

    def sensible_columns(self):
        return ["sex"]

    def y_columns(self):
        return ["y"]

    def dataset_path(self):
        return '%s/default.csv' % (self.workingdir)

    def sep(self):
        return ','

    def files_to_retrieve(self):
        return [('https://datacloud.di.unito.it/index.php/s/qCg4Qd4RKEyp9ap/download',
                '%s/default.csv' % (self.workingdir))]

    def prepare_all(self):
        if os.path.isfile(self.dataset_path()):
            logging.info("Default datafile already exists. Using existing version.")
            return

        ds = pandas.read_csv(DefaultDataset.DOWNLOAD_PATH, sep=',')

        ds.columns = self.all_columns()
        ds.to_csv(self.dataset_path(), index=False)
        