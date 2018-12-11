from .dataset_base import DatasetBase
import pandas
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import numpy as np


class SynthEasyDataset(DatasetBase):
    def name(self):
        return "SynthEasy"

    def all_columns(self):
        return ["x","s", "y"]

    def one_hot_columns(self):
        return ["s", "y"]

    def sensible_columns(self):
        return ["s"]

    def y_columns(self):
        return ["y"]

    def dataset_path(self):
        return 'data/synth-easy.csv'

    def sep(self):
        return ','

    def files_to_retrieve(self):
        return []

    def prepare_all(self):
        pass
