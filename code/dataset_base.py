import csv
import requests
import os.path
import tensorflow as tf
import numpy as np
import zipfile
import pandas
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from tqdm import tqdm


class DatasetBase:
    def __init__(self):
        """
        Downloads and load into memory the dataset.
        """
        self.download_all()
        self.prepare_all()
        self.load_all()

    def prepare_all(self):
        pass

    def all_columns(self):
        pass

    def one_hot_columns(self):
        pass

    def sensible_columns(self):
        pass

    def num_s_columns(self):
        pass

    def num_y_columns(self):
        return 2

    def dataset_path(self):
        pass

    def sep(self):
        return ';'

    def files_to_retrieve(self):
        pass

    def load_data(self):
        """
        Loads the file specified by the path parameter, parses it
        and returns three lists containing the resulting examples, labels and secret variables (xs,ys,s).
        In order for the method to work properly each column name must *not* be a prefix of another column
        name (indeed this must be true for the sensible and the class columns, but it would be indeed better
        to enforce the constraint on all columns).
        """

        print("Importing %s" % (self.dataset_path()))
        dataset = pandas.read_csv(self.dataset_path(), sep=self.sep())
        s_col_names = self.sensible_columns()
        y_col_names = self.y_columns()

        assert len(s_col_names) <= 1, "multiple s columns not yet supported"
        assert len(y_col_names) <= 1, "multiple y columns not yet supported"

        df = pandas.get_dummies(dataset, columns=self.one_hot_columns())
        non_hot_cols = [col for col in self.all_columns() if col not in self.one_hot_columns()]

        scaler = MinMaxScaler()
        df[non_hot_cols] = scaler.fit_transform(df[non_hot_cols])

        s_1h_col_names = df.columns[[colname for s_col_name in s_col_names for colname in df.columns.str.startswith(s_col_name)]]
        y_1h_col_names = df.columns[[colname for y_col_name in y_col_names for colname in df.columns.str.startswith(y_col_name)]]
        all_non_y_names = [col for col in df.columns if col not in y_1h_col_names]

        self._num_s_columns = len(s_1h_col_names)
        self._num_y_columns = len(y_1h_col_names)

        print("all non y names: %s" % (','.join(all_non_y_names)))
        print("all y names: %s" % (','.join(y_1h_col_names)))
        print("all s names: %s" % (','.join(s_1h_col_names)))

        xs = df[all_non_y_names]
        ys = df[y_1h_col_names]
        s = df[s_1h_col_names]

        return (xs.as_matrix(),ys.as_matrix(),s.as_matrix())

    def num_s_columns(self):
        return self._num_s_columns

    def num_y_columns(self):
        return self._num_y_columns

    def sample_examples(self, xs, ys, class_vec, num_elems):
        class_examples = np.where(ys == class_vec)[0]
        extracted = np.random.choice(class_examples, num_elems, replace=True)

        return (xs[extracted], ys[extracted])

    def oversample_dataset(self, data):
        xs,ys = data
        xs, ys = np.array(xs), np.array(ys)
        neg_count = np.count_nonzero(ys[:,0])
        pos_count = np.count_nonzero(ys[:,1])

        diff_count = pos_count - neg_count

        if diff_count == 0:
            return

        if diff_count > 0:
            (sampled_xs, sampled_ys) = self.sample_examples(xs, ys, [1.0, 0.0], diff_count)
        else:
            (sampled_xs, sampled_ys) = self.sample_examples(xs, ys, [0.0, 1.0], -diff_count)

        return (list(xs) + list(sampled_xs), list(ys) + list(sampled_ys))

    def undersample_dataset(self,dataset):
        neg_indexes = np.where( dataset[1][:,0] == 1 )[0]
        pos_indexes = np.where( dataset[1][:,1] == 1 )[0]
        sampled_indexes = np.random.choice(neg_indexes, len(pos_indexes), replace=False)

        newxs = np.vstack([dataset[0][pos_indexes], dataset[0][sampled_indexes]])
        newys = np.vstack([dataset[1][pos_indexes], dataset[1][sampled_indexes]])

        return (newxs, newys)

    def load_all(self):
        """
        loads into memory the training and the test sets (it needs to
        be called before accessing to them using other methods that
        access to the train and the test set)
        """
        xs,ys,s = self.load_data()

        train_xs, test_xs, train_ys, test_ys, train_s, test_s = train_test_split(xs,ys,s,test_size=0.1, random_state=42)

        self._traindata = (train_xs, train_ys, train_s)
        self._testdata = (test_xs, test_ys, test_s)

        self._train_dataset = tf.data.Dataset.from_tensor_slices(self._traindata)
        self._test_dataset = tf.data.Dataset.from_tensor_slices(self._testdata)

        self.print_datasets_stats([("Train", self._traindata), ("Test", self._testdata)])


    def download(self, url, filename):
        """
        downloads the file pointed by the given url and saves it using
        the given filename
        """
        if os.path.isfile(filename):
            return

        dataset = requests.get(url)

        print("Downloading %s" % (url))
        with open(filename, 'wb') as file:
            for data in tqdm(dataset):
                file.write(data)

    def print_columns_stats(self):
        print("|Dataset|    count|  count +|  count -|     min |     max |")
        print("|:-----:|--------:|--------:|--------:|--------:|--------:|")

    def print_datasets_stats(self, datasets):
        self.print_columns_stats()

        for (name, dataset) in datasets:
            pos_set_len = len(np.where(dataset[1][:,1] == 1.0)[0])
            neg_set_len = len(np.where(dataset[1][:,0] == 1.0)[0])
            min_col_val = dataset[0].min()
            max_col_val = dataset[0].max()
            print("|%7s|%9d|%9d|%9d|%9.5f|%9.4f|" % (name, len(dataset[1]), pos_set_len, neg_set_len, min_col_val, max_col_val))

        print("")


    def download_all(self):
        """
        download the trainig set and the test set if needed
        """

        for (dataurl, filepath) in self.files_to_retrieve():
            self.download( dataurl, filepath)

    def num_features(self):
        return self._traindata[0].shape[1]

    def train_dataset(self):
        """
        returns a tf.data.Dataset built from the training set
        """
        return self._train_dataset

    def test_dataset(self):
        """
        returns a tf.data.Dataset built from the test set
        """
        return self._test_dataset

    def train_all_data(self):
        """
        returns the whole training set as a tuple of numpy arrays (xs,ys,s)
        """
        xs,ys,s = self._traindata
        return (np.array(xs), np.array(ys), np.array(s))

    def test_all_data(self):
        """
        returns the whole test set as a tuple of numpy arrays (xs,ys,s)
        """
        xs,ys,s = self._testdata
        return (np.array(xs), np.array(ys), np.array(s))