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

    def __init__(self):
        """
        Downloads and load into memory the dataset. If balance_trainset is True then
        the negative examples are oversampled so to get a 50/50 split between the positive
        and the negative class.
        """
        self.download_all()
        self.load_all()

    def all_columns(self):
        return ["age","job","marital","education","default","balance","housing","loan","contact","day","month","duration","campaign","pdays","previous","poutcome"]

    def one_hot_columns(self):
        return ["job","marital","education","default", "housing","loan","contact","poutcome","month","y"]

    def dataset_path(self):
        return 'data/bank-full.csv'

    def sep(self):
        return ';'

    def files_to_retrieve(self):
        return [('https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank.zip',
                'data/bank.zip')]

    def prepare_data(self):
        zip_ref = zipfile.ZipFile('data/bank.zip', 'r')
        zip_ref.extractall('data/')
        zip_ref.close()
