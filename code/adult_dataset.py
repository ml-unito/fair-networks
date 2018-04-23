import csv
import requests
import os.path
import tensorflow as tf
import numpy as np

from tqdm import tqdm


class AdultDataset:
    """
    Helper class allowing to download and load into memory the adult dataset
    """
    TRAINURL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'
    TESTURL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test'
    TRAINPATH = 'data/adult.data'
    TESTPATH = 'data/adult.test'


    WORKCLASS = ['Private', 'Self-emp-not-inc',  'Self-emp-inc', 'Federal-gov', 'Local-gov', 'State-gov', 'Without-pay', 'Never-worked', '?']
    EDUCATION = [ 'Bachelors', 'Some-college', '11th', 'HS-grad', 'Prof-school', 'Assoc-acdm', 'Assoc-voc', '9th', '7th-8th', '12th', 'Masters', '1st-4th', '10th', 'Doctorate', '5th-6th', 'Preschool']
    MARITAL_STATUS = [ 'Married-civ-spouse', 'Divorced', 'Never-married', 'Separated', 'Widowed', 'Married-spouse-absent', 'Married-AF-spouse']
    OCCUPATION = [ 'Tech-support', 'Craft-repair', 'Other-service', 'Sales', 'Exec-managerial', 'Prof-specialty', 'Handlers-cleaners', 'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing', 'Transport-moving', 'Priv-house-serv', 'Protective-serv', 'Armed-Forces','?']
    RELATIONSHIP = [ 'Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative', 'Unmarried']
    RACE = ['White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other', 'Black']
    SEX = ['Female', 'Male']
    NATIVE_COUNTRY = [ 'United-States', 'Cambodia', 'England', 'Puerto-Rico', 'Canada', 'Germany', 'Outlying-US(Guam-USVI-etc)', 'India', 'Japan', 'Greece', 'South', 'China', 'Cuba', 'Iran', 'Honduras', 'Philippines', 'Italy', 'Poland', 'Jamaica', 'Vietnam', 'Mexico', 'Portugal', 'Ireland', 'France', 'Dominican-Republic', 'Laos', 'Ecuador', 'Taiwan', 'Haiti', 'Columbia', 'Hungary', 'Guatemala', 'Nicaragua', 'Scotland', 'Thailand', 'Yugoslavia', 'El-Salvador', 'Trinadad&Tobago', 'Peru', 'Hong', 'Holand-Netherlands', '?']
    LABELS = ['>50K', '<=50K']

    def __init__(self, balance_trainset=True):
        """
        Downloads and load into memory the dataset. If balance_trainset is True then
        the negative examples are oversampled so to get a 50/50 split between the positive
        and the negative class.
        """
        self.balance_trainset = balance_trainset
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

    def mapline_simple(self, line):
        if line == [] or line[0][0] == '|':
            return ([],[])

        result = []

        result.append([int(line[0])])
        result.append([line[1]])
        result.append([int(line[2])])
        result.append([line[3]])
        result.append([int(line[4])])
        result.append([line[5]])
        result.append([line[6]])
        result.append([line[7]])
        result.append([line[8]])
        result.append([line[9]])
        result.append([int(line[10])])
        result.append([int(line[11])])
        result.append([int(line[12])])
        result.append([line[13]])

        labels = self.remove_dot(line[14])
        labels = self.one_hot(labels, self.LABELS)

        return (sum(result, []), labels)


    def mapline_one_hot(self, line):
        """
        given a vector of string attributes read from the adult dataset maps
        them into numeric features (using one_hot_encoding for categorical attributes).
        It returs a pair (x,y) where x is the encoding of the example and y is the
        associated label.
        """
        if line == [] or line[0][0] == '|':
            return ([],[])

        result = []

        result.append([int(line[0])])
        result.append(self.one_hot(line[1], self.WORKCLASS))
        result.append([int(line[2])])
        # result.append(self.one_hot(line[3], self.EDUCATION))
        result.append([int(line[4])])
        result.append(self.one_hot(line[5], self.MARITAL_STATUS))
        result.append(self.one_hot(line[6], self.OCCUPATION))
        result.append(self.one_hot(line[7], self.RELATIONSHIP))
        result.append(self.one_hot(line[8], self.RACE))
        result.append(self.one_hot(line[9], self.SEX))
        result.append([int(line[10])])
        result.append([int(line[11])])
        result.append([int(line[12])])
        result.append(self.one_hot(line[13], self.NATIVE_COUNTRY))

        labels = self.remove_dot(line[14])
        labels = self.one_hot(labels, self.LABELS)

        return (sum(result, []), labels)

    def load_data(self, path):
        """
        Loads the file specified by the path parameter, parses it
        according to the Adult file format and returns a pair of
        lists containing the resulting examples and labels (xs,ys)
        """
        print("Importing %s" % (path))
        num_lines = num_lines = sum(1 for line in open(path))
        xs = []
        ys = []

        with open(path,'r') as file:
            reader = csv.reader(file, skipinitialspace=True)
            for line in tqdm(reader, total=num_lines):
                x,y  = self.mapline_one_hot(line)
                if x != []:
                    xs.append(x)
                    ys.append(y)

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
        self._traindata = self.balance(self.load_data(self.TRAINPATH))
        self._testdata = self.load_data(self.TESTPATH)

        print(np.count_nonzero(np.array(self._traindata[1])[:,0]))
        print(np.count_nonzero(np.array(self._traindata[1])[:,1]))
        print("|Train| = %d" % len(self._traindata[0]))
        print("|Test| = %d" % len(self._testdata[0]))

        train_xs = np.array(self._traindata[0])
        train_ys = np.array(self._traindata[1])
        test_xs = np.array(self._testdata[0])
        test_ys = np.array(self._testdata[1])

        self._train_dataset = tf.data.Dataset.from_tensor_slices((train_xs, train_ys))
        self._test_dataset = tf.data.Dataset.from_tensor_slices((test_xs, test_ys))
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
        self.download( self.TRAINURL, self.TRAINPATH)
        self.download( self.TESTURL, self.TESTPATH)

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