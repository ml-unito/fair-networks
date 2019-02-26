from .dataset_base import DatasetBase
import pandas
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import numpy as np


class SynthEasy4Dataset(DatasetBase):
    def name(self):
        return "SynthEasy4"

    def all_columns(self):
        return ["x1", "x2", "hidden_s", "s", "y"]

    def one_hot_columns(self):
        return ["hidden_s", "s", "y"]

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
