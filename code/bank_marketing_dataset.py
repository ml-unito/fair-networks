import csv
import requests
import os.path
import tensorflow as tf
import numpy as np
import zipfile
import pandas
import sklearn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from dataset_base import DatasetBase

from tqdm import tqdm


class BankMarketingDataset(DatasetBase):
    """
    Helper class allowing to download and load into memory the adult dataset
    """

    def name(self):
        return "Bank"

    def all_columns(self):
        return ["age","job","marital","education","default","balance","housing","loan","contact","day","month","duration","campaign","pdays","previous","poutcome"]

    def one_hot_columns(self):
        return ["job","marital","education","default", "housing","loan","contact","poutcome","month","y"]

    def sensible_columns(self):
        return ["marital"]

    def y_columns(self):
        return ["y"]

    def dataset_path(self):
        return '%s/bank-full.csv' % (self.workingdir)

    def sep(self):
        return ';'

    def files_to_retrieve(self):
        return [('https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank.zip',
                '%s/bank.zip' % (self.workingdir))]

    def prepare_all(self):
        if os.path.isfile(self.dataset_path()):
            print("Bank datafile already exists. Using existing version.")
            return

        zip_ref = zipfile.ZipFile('%s/bank.zip' % (self.workingdir), 'r')
        zip_ref.extractall(self.workingdir)
        zip_ref.close()
