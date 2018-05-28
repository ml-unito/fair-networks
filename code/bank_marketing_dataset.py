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
    def all_columns(self):
        return ["age","job","marital","education","default","balance","housing","loan","contact","day","month","duration","campaign","pdays","previous","poutcome"]

    def one_hot_columns(self):
        return ["job","marital","education","default", "housing","loan","contact","poutcome","month","y"]

    def sensible_columns(self):
        return ["marital"]

    def y_columns(self):
        return ["y"]

    def dataset_path(self):
        return 'data/bank-full.csv'

    def sep(self):
        return ';'

    def files_to_retrieve(self):
        return [('https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank.zip',
                'data/bank.zip')]

    def prepare_all(self):
        zip_ref = zipfile.ZipFile('data/bank.zip', 'r')
        zip_ref.extractall('data/')
        zip_ref.close()
