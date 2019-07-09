import csv
import requests
import os.path
import tensorflow as tf
import numpy as np
import zipfile
import pandas
import sklearn
import logging

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from fair.datasets.dataset_base import DatasetBase

from tqdm import tqdm


class FakeNewsDataset(DatasetBase):
    DOWNLOAD_PATH = 'data/fakenews.csv'

    """
    Helper class allowing to download and load into memory the fake news dataset
    """
    def __init__(self, workingdir, existing_split=False):
        """
        Downloads and load into memory the dataset.
        """
        self.workingdir = workingdir
        self.download_all()
        self.prepare_all()
        self.load_all()

    def name(self):
        return "Default"

    def all_columns(self):
        l = ['f'+str(i) for i in range(0, 33334)]
        return l + ["dominio",  "y"]

    def one_hot_columns(self):
        return ["dominio", "y"]

    def sensible_columns(self):
        return ["dominio"]

    def y_columns(self):
        return ["y"]

    def dataset_path(self):
        return '%s/fakenews.csv' % (self.workingdir)

    def sep(self):
        return ','

    def files_to_retrieve(self):
        return [('https://datacloud.di.unito.it/index.php/s/mTrCZn2C2WXBMB4/download',
                '%s/fakenews.csv' % (self.workingdir))]

    def prepare_all(self):
        pass
        #if os.path.isfile(self.dataset_path()):
        #    logging.info("FakeNews datafile already exists. Using existing version.")
        #    return

        #ds = pandas.read_csv(DefaultDataset.DOWNLOAD_PATH, sep=',')

        #ds.columns = self.all_columns()
        #ds.to_csv(self.dataset_path(), index=False)
 