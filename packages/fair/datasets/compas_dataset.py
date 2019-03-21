import csv
import requests
import os.path
import tensorflow as tf
import numpy as np
import logging

from tqdm import tqdm

from .dataset_base import DatasetBase

class CompasDataset(DatasetBase):
    """
    Helper class allowing to download and load into memory the compas dataset
    """
    def name(self):
        return "Compas"

    def all_columns(self):
        return ["race", "sex", "priors_count", "age_cat", "c_charge_degree", "two_year_recid"]

    def one_hot_columns(self):
        return ["race", "sex", "age_cat", "c_charge_degree", "two_year_recid"]

    def dataset_path(self):
        return "%s/compas-violent-preprocessed.csv" % (self.workingdir)

    def sep(self):
        return ','

    def files_to_retrieve(self):
        return [
            ('https://datacloud.di.unito.it/index.php/s/Y6xd9AtKC8bnPtc/download',
             '%s/compas-violent-preprocessed.data' % self.workingdir),
        ]

    def sensible_columns(self):
        return["race"]

    def y_columns(self):
        return ["two_year_recid"]

    def prepare_all(self):
        if os.path.isfile(self.dataset_path()):
            logging.info("Compas dataset already exist. Using existing version.")
            return
        

