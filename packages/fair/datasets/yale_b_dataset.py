import csv
import requests
import os.path
import tensorflow as tf
import numpy as np
import pandas
import logging
from sklearn.preprocessing import MinMaxScaler


from tqdm import tqdm

from .dataset_base import DatasetBase

class YaleBDataset(DatasetBase):
    """
    Helper class allowing to load into memory the Yale Faces B dataset
    """
    
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

    def test_path(self):
        return "%s/yale_test.csv" % (self.workingdir)

    def train_path(self):
        return "%s/yale_train.csv" % (self.workingdir)

    def sep(self):
        return ','

    def files_to_retrieve(self):
        pass

    def sensible_columns(self):
        return ['s']

    def y_columns(self):
        return ['y']

    def prepare_all(self):
        pass

    def train_path(self):
        return "%s/yale_train.csv" % (self.workingdir)

    def test_path(self):
        return "%s/yale_test.csv"  % (self.workingdir)

    def prepare_all(self):
        pass

    def download_all(self):
        pass

    def load_all(self):
        """
        An alternative to load_all where the dataset has already been separated into 
        train and test files.
        """
        (train_xs, train_ys, train_s), (test_xs, test_ys,
                                        test_s) = self.load_data_separate_paths(self.train_path(), self.test_path())

        self._traindata = (train_xs, train_ys, train_s)
        self._testdata = (test_xs, test_ys, test_s)

        self._train_dataset = tf.data.Dataset.from_tensor_slices(
            self._traindata)
        self._test_dataset = tf.data.Dataset.from_tensor_slices(self._testdata)



    def load_data_separate_paths(self, train_path, test_path):
        """
        Loads the file specified by the path parameter, parses it
        and returns three lists containing the resulting examples, labels and secret variables (xs,ys,s).
        In order for the method to work properly each column name must *not* be a prefix of another column
        name (indeed this must be true for the sensible and the class columns, but it would be indeed better
        to enforce the constraint on all columns).
        """
        logging.info("Reading dataset: {}".format(self.dataset_path()))

        train_dataset = pandas.read_csv(train_path, sep=self.sep())
        test_dataset = pandas.read_csv(test_path, sep=self.sep())
        num_train_examples = train_dataset.shape[0]
        dataset = pandas.concat([train_dataset, test_dataset], axis=0)

        logging.debug("[LOAD BEGIN] y sums:\n {}".format(
            dataset.groupby(self.y_columns()).nunique()))

        s_col_names = self.sensible_columns()
        y_col_names = self.y_columns()

        assert len(s_col_names) <= 1, "multiple s columns not yet supported"
        assert len(y_col_names) <= 1, "multiple y columns not yet supported"

        logging.info("Getting dummy variables for columns: {}".format(
            list(self.one_hot_columns())))

        df = pandas.get_dummies(dataset, columns=self.one_hot_columns())
        non_hot_cols = [col for col in self.all_columns() if col not in self.one_hot_columns()]
        train_dataset = df.iloc[:num_train_examples]
        test_dataset = df.iloc[num_train_examples:]

        logging.info("Scaling values for columns: {}".format(list(non_hot_cols)))


        scaler = MinMaxScaler()
        train_dataset[non_hot_cols] = scaler.fit_transform(train_dataset[non_hot_cols].astype(np.float64))
        test_dataset[non_hot_cols] = scaler.transform(test_dataset[non_hot_cols].astype(np.float64))

        s_1h_col_names = df.columns[[colname for s_col_name in s_col_names for colname in df.columns.str.startswith(s_col_name)]]
        y_1h_col_names = df.columns[[colname for y_col_name in y_col_names for colname in df.columns.str.startswith(y_col_name)]]
        all_non_y_non_s_names = [col for col in df.columns if col not in y_1h_col_names and col not in s_1h_col_names]

        self._num_s_columns = len(s_1h_col_names)
        self._num_y_columns = len(y_1h_col_names)

        xs = train_dataset[all_non_y_non_s_names]
        ys = train_dataset[y_1h_col_names]
        s = train_dataset[s_1h_col_names]
        
        xt = test_dataset[all_non_y_non_s_names]
        yt = test_dataset[y_1h_col_names]
        st = test_dataset[s_1h_col_names]

        logging.debug("y values sums:{}".format(list(ys.sum(axis=0))))
        logging.debug("s values sums:{}".format(list(s.sum(axis=0))))

        return (xs.values, ys.values, s.values), (xt.values, yt.values, st.values)
        
