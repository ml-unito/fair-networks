from .dataset_base import DatasetBase
import pandas
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow.compat.v1 as tf
import numpy as np


class SynthEasy3Dataset(DatasetBase):
    def name(self):
        return "SynthEasy3"

    def all_columns(self):
        return ["x1", "x2", "s", "y"]

    def one_hot_columns(self):
        return ["s", "y"]

    def sensible_columns(self):
        return ["s"]

    def y_columns(self):
        return ["y"]

    def dataset_path(self):
        return 'data/synth-easy3.csv'

    def sep(self):
        return ','

    def files_to_retrieve(self):
        return []

    def prepare_all(self):
        pass
