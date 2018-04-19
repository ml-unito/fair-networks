# class AdultDataset:
#     PATH='/data/adultdata'

import csv
import requests
import os.path
import tensorflow as tf
from tqdm import tqdm


class AdultDataset:
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

    def __init__(self):
        self.download_all()
        self.load_all()

    def one_hot(self, value, dictionary):
        vector = [0 for x in range(len(dictionary))]
        vector[ dictionary.index(value) ] = 1.0
        return vector

    def remove_dot(self, line):
        """
        Remove a dot at the end of the line.
        This function is needed because lines in the test set
        ends with a '.' (this is not true of lines in the training set)
        """
        if line[-1] == '.':
            return line[0:-1]
        else:
            return line

    def mapline(self, line):
        if line == [] or line[0][0] == '|':
            return ([],[])

        result = []

        result.append([int(line[0])])
        result.append(self.one_hot(line[1], self.WORKCLASS))
        result.append([int(line[2])])
        result.append(self.one_hot(line[3], self.EDUCATION))
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

        label = self.remove_dot(line[14])
        labels = self.one_hot(label, self.LABELS)

        return (sum(result, []), labels)

    def load_dataset(self, path):
        print("Importing %s" % (path))
        num_lines = num_lines = sum(1 for line in open(path))
        xs = []
        ys = []

        with open(path,'r') as file:
            reader = csv.reader(file, skipinitialspace=True)
            for line in tqdm(reader, total=num_lines):
                x,y  = self.mapline(line)
                if x != []:
                    xs.append(x)
                    ys.append(y)

        return tf.data.Dataset.from_tensor_slices((xs, ys))

    def load_all(self):
        self.traindataset = self.load_dataset(self.TRAINPATH)
        self.testdataset = self.load_dataset(self.TESTPATH)


    def download(self, url, filename):
        if os.path.isfile(filename):
            return

        dataset = requests.get(url)

        print("Downloading %s" % (url))
        with open(filename, 'wb') as file:
            for data in tqdm(dataset):
                file.write(data)

    def download_all(self):
        self.download( self.TRAINURL, self.TRAINPATH)
        self.download( self.TESTURL, self.TESTPATH)

    def traindata(self):
        return self.traindataset

    def testdata(self):
        return self.testdataset
