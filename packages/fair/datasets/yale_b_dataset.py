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

    def column_indices(self, df, cols):
        return [df.columns.get_loc(col) for col in cols]

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
        non_hot_cols_indices = self.column_indices(df, non_hot_cols)

        train_dataset_non_hot = df.iloc[:num_train_examples, non_hot_cols_indices]
        test_dataset_non_hot = df.iloc[num_train_examples:, non_hot_cols_indices]

        scaler = MinMaxScaler()
        df.iloc[:num_train_examples, non_hot_cols_indices] = scaler.fit_transform(train_dataset_non_hot.astype(np.float64))
        df.iloc[num_train_examples:, non_hot_cols_indices] = scaler.transform(test_dataset_non_hot.astype(np.float64))

        s_1h_col_names = df.columns[[colname for s_col_name in s_col_names for colname in df.columns.str.startswith(s_col_name)]]
        y_1h_col_names = df.columns[[colname for y_col_name in y_col_names for colname in df.columns.str.startswith(y_col_name)]]
        all_non_y_non_s_names = [col for col in df.columns if col not in y_1h_col_names and col not in s_1h_col_names]
        s_1h_col_names_indices = self.column_indices(df, s_1h_col_names)
        y_1h_col_names_indices = self.column_indices(df, y_1h_col_names)
        all_non_y_non_s_names_indices = self.column_indices(df, all_non_y_non_s_names)

        logging.debug("s len:{} y len:{} other len:{}".format(len(s_1h_col_names), len(y_1h_col_names), len(all_non_y_non_s_names)))

        self._num_s_columns = len(s_1h_col_names)
        self._num_y_columns = len(y_1h_col_names)

        xs = df.iloc[:num_train_examples, all_non_y_non_s_names_indices].values
        ys = df.iloc[:num_train_examples, y_1h_col_names_indices].values
        s =  df.iloc[:num_train_examples, s_1h_col_names_indices].values
        
        xt = df.iloc[num_train_examples:, all_non_y_non_s_names_indices].values
        yt = df.iloc[num_train_examples:, y_1h_col_names_indices].values
        st = df.iloc[num_train_examples:, s_1h_col_names_indices].values

        logging.debug("xs sample: {}".format(xs[:10]))
        logging.debug("ys sample: {}".format(ys[:10]))
        logging.debug("s sample: {}".format(s[:10]))


        return (xs, ys, s), (xt, yt, st)
        
