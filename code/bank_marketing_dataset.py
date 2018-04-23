import csv
import requests
import os.path
import tensorflow as tf
import numpy as np
import zipfile
import pandas
import sklearn

from tqdm import tqdm


class BankMarketingDataset:
    """
    Helper class allowing to download and load into memory the adult dataset
    """
    DATAURL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank.zip'
    # TESTURL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test'

    ZIPPATH = 'data/bank.zip'
    DATADIR = 'data/'
    DATAPATH = 'data/bank-full.csv'
    # TESTPATH = 'data/bank_marketing.test'

    ONE_HOT_COLUMNS = ["job","marital","education","default", "housing","loan","contact","poutcome","month","y"]


    def __init__(self):
        """
        Downloads and load into memory the dataset. If balance_trainset is True then
        the negative examples are oversampled so to get a 50/50 split between the positive
        and the negative class.
        """
        self.download_all()
        self.load_all()

    def one_hot(self, value, dictionary):
        """
        Returns a one-hot-encoding of the given value. The hot encoding is based
        on the list of possibible values specified by the dictionary parameter.
        """
        vector = [0 for x in range(len(dictionary))]
        vector[ dictionary.index(value) ] = 1.0
        return vector

    def remove_dot(self, line):
        """
        Removes a dot at the end of the line.
        This function is needed because lines in the test set
        ends with a '.' (this is not true of lines in the training set)
        """
        if line[-1] == '.':
            return line[0:-1]
        else:
            return line


    def load_data(self, path):
        """
        Loads the file specified by the path parameter, parses it
        according to the Adult file format and returns a pair of
        lists containing the resulting examples and labels (xs,ys)
        """

        print("Importing %s" % (path))
        dataset = pandas.read_csv(path, sep=';')
        df = pandas.get_dummies(dataset, columns=self.ONE_HOT_COLUMNS)



        matrix = df.as_matrix()
        xs = matrix[:,:-2]
        ys = matrix[:, -2:]

        return (xs,ys)


    def sample_examples(self, xs, ys, class_vec, num_elems):
        class_examples = np.where(ys == class_vec)[0]
        extracted = np.random.choice(class_examples, num_elems, replace=True)

        return (xs[extracted], ys[extracted])


    def balance(self, data):
        if not self.balance_trainset:
            return data

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


    def load_all(self):
        """
        loads into memory the training and the test sets (it needs to
        be called before accessing to them using other methods that
        access to the train and the test set)
        """
        xs,ys = self.load_data(self.DATAPATH)
        train_xs, test_xs, train_ys, test_ys = sklearn.model_selection.train_test_split(xs,ys,test_size=0.3)

        self._traindata = (train_xs, train_ys)
        self._testdata = (test_xs, test_ys)

        # self._testdata = self.load_data(self.TESTPATH)

        # print(np.count_nonzero(np.array(self._traindata[1])[:,0]))
        # print(np.count_nonzero(np.array(self._traindata[1])[:,1]))
        # print("|Train| = %d" % len(self._traindata[0]))
        # print("|Test| = %d" % len(self._testdata[0]))

        # test_xs = np.array(self._testdata[0])
        # test_ys = np.array(self._testdata[1])

        self._train_dataset = tf.data.Dataset.from_tensor_slices(self._traindata)
        self._test_dataset = tf.data.Dataset.from_tensor_slices(self._testdata)
        print("Imported")

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

    def download_all(self):
        """
        download the trainig set and the test set if needed
        """
        self.download( self.DATAURL, self.ZIPPATH)
        zip_ref = zipfile.ZipFile(self.ZIPPATH, 'r')
        zip_ref.extractall(self.DATADIR)
        zip_ref.close()

        # self.download( self.TESTURL, self.TESTPATH)

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
        returns the whole training set as a pair of numpy arrays (xs,ys)
        """
        xs,ys = self._traindata
        return (np.array(xs), np.array(ys))

    def test_all_data(self):
        """
        returns the whole test set as a pair of numpy arrays (xs,ys)
        """
        xs,ys = self._testdata
        return (np.array(xs), np.array(ys))
