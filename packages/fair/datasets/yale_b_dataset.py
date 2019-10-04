import csv
import requests
import os.path
import tensorflow.compat.v1 as tf
import numpy as np
import pandas as pd
import logging
import tarfile
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

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
        return [ 
            ("https://datacloud.di.unito.it/index.php/s/ZNE76RWzNw2YQtp/download",
             "{}/yale_dataset.tar.gz".format(self.workingdir))
            ]

    def sensible_columns(self):
        return ['s']

    def y_columns(self):
        return ['y']

    def _load_pickle(self, path):
        logging.info("Reading pickle file: {}".format(path))

        with open(path, 'rb') as f:
            data_dict = pickle.load(f, encoding='latin1')

        x = pd.DataFrame(data_dict['x'])
        y = pd.DataFrame(data_dict['t'])
        s = pd.DataFrame(data_dict['light'])
        y = y.rename({0: 'y'}, axis=1)
        s = s.rename({0: 's'}, axis=1)
        ext = range(0, 504)
        x = x.rename({e: 'x_' + str(e) for e in ext}, axis=1)
        return x, y, s

    def _pickle_files(self):
        files = ['{}/set_{}.pdata'.format(self.workingdir, i) for i in range(0, 5)]
        test_file = '{}/test.pdata'.format(self.workingdir)
        return (files, test_file)

    def _process_pickle_files(self):
        if os.path.isfile(self.train_path()) and os.path.isfile(self.test_path()):
            logging.info("Dataset already exists...")
            return

        logging.info("Processing pickle files...")

        files, test_file = self._pickle_files()

        x = pd.DataFrame([])
        y = pd.DataFrame([])
        s = pd.DataFrame([])

        for path in files:
            tx, ty, ts = self._load_pickle(path)
            x = pd.concat([x, tx], axis=0)
            y = pd.concat([y, ty], axis=0)
            s = pd.concat([s, ts], axis=0)

        y = pd.DataFrame(y)
        s = pd.DataFrame(s)
        train = pd.concat([x, y, s], axis=1)

        xt, yt, st = self._load_pickle(test_file)
        test = pd.concat([xt, yt, st], axis=1)
        test = test[test['s'] < 5]

        train.to_csv('{}/yale_train.csv'.format(self.workingdir), index=False)
        test.to_csv('{}/yale_test.csv'.format(self.workingdir), index=False)

    def _extract_pickles(self):
        tarfile_path = '{}/yale_dataset.tar.gz'.format(self.workingdir)
        train, test = self._pickle_files()
        train_ok = not (False in [os.path.isfile(file) for file in train])
        test_ok = os.path.isfile(test)

        if train_ok and test_ok:
            logging.info(
                "All train and test pickle exist skipping untaring them")
            return

        logging.info("Extracting yale dataset from: {}".format(tarfile_path))

        tar = tarfile.open(name=tarfile_path, mode="r|gz")
        tar.extractall(path=self.workingdir)
        tar.close()

    def prepare_all(self):

        self._extract_pickles()
        self._process_pickle_files()

    def train_path(self):
        return "%s/yale_train.csv" % (self.workingdir)

    def test_path(self):
        return "%s/yale_test.csv"  % (self.workingdir)

    def load_all(self):
        """
        An alternative to load_all where the dataset has already been separated into 
        train and test files.
        """
        (train_xs, train_ys, train_s), (val_xs, val_ys,val_s) = self.load_data_separate_paths(self.train_path(), self.test_path())

        # train_selector = (np.argmax(train_s, axis=1) != 4)
        # val_selector = (np.argmax(train_s, axis=1) == 4)

        # train_xs, val_xs = train_xs[train_selector], train_xs[val_selector]
        # train_ys, val_ys = train_ys[train_selector], train_ys[val_selector]
        # train_s, val_s = train_s[train_selector], train_s[val_selector]

        test_xs, test_ys, test_s = (np.zeros([1, train_xs.shape[1]]),
                                  np.zeros([1, train_ys.shape[1]]),
                                  np.zeros([1, train_s.shape[1]]))

        # train_xs, val_xs, train_ys, val_ys, train_s, val_s = train_test_split(train_xs, train_ys, train_s, test_size=0.2)
        
        self._traindata = (train_xs, train_ys, train_s)
        self._valdata = (val_xs, val_ys, val_s)
        self._testdata = (test_xs, test_ys, test_s)

        self._train_dataset = tf.data.Dataset.from_tensor_slices(self._traindata)
        self._val_dataset = tf.data.Dataset.from_tensor_slices(self._valdata)
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
        logging.info("Reading datasets: {} and {}".format(train_path, test_path))

        train_dataset = pd.read_csv(train_path, sep=self.sep())
        test_dataset = pd.read_csv(test_path, sep=self.sep())
        num_train_examples = train_dataset.shape[0]
        dataset = pd.concat([train_dataset, test_dataset], axis=0)

        logging.debug("[LOAD BEGIN] y sums:\n {}".format(
            dataset.groupby(self.y_columns()).nunique()))

        s_col_names = self.sensible_columns()
        y_col_names = self.y_columns()

        assert len(s_col_names) <= 1, "multiple s columns not yet supported"
        assert len(y_col_names) <= 1, "multiple y columns not yet supported"

        logging.info("Getting dummy variables for columns: {}".format(
            list(self.one_hot_columns())))

        df = pd.get_dummies(dataset, columns=self.one_hot_columns())
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
        
