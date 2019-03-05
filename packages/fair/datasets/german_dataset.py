from .dataset_base import DatasetBase
import pandas
import os.path
import logging

class GermanDataset(DatasetBase):
    DOWNLOAD_PATH = 'data/german-original.data'

    """
    Helper class allowing to download and load into memory the german dataset
    """

    def name(self):
        return "German"

    def all_columns(self):
        return [ "Attribute_1", "Attribute_2", "Attribute_3", "Attribute_4", "Attribute_5", "Attribute_6", "Attribute_7", "Attribute_8", "Gender", "Attribute_10", "Attribute_11", "Attribute_12", "Attribute_13", "Attribute_14", "Attribute_15", "Attribute_16", "Attribute_17", "Attribute_18", "Attribute_19", "Attribute_20", "y" ]

    def one_hot_columns(self):
        return [ "Attribute_1", "Attribute_3", "Attribute_4", "Attribute_6", "Attribute_7", "Gender", "Attribute_10", "Attribute_12", "Attribute_14", "Attribute_15", "Attribute_17", "Attribute_19", "Attribute_20", "y" ]

    def dataset_path(self):
        return "%s/german-train-test.csv" % (self.workingdir)

    def sep(self):
        return ','

    def files_to_retrieve(self):
        return [
            ('https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data',
             self.DOWNLOAD_PATH),
        ]

    def sensible_columns(self):
        return["Gender"]

    def y_columns(self):
        return ["y"]

    def prepare_all(self):
        if os.path.isfile(self.dataset_path()):
            logging.info("German dataset already exist. Using existing version.")
            return

        ds = pandas.read_csv(GermanDataset.DOWNLOAD_PATH, sep=' ', header=None)
        # replacing column 8 with 'gender' -- as far as we can tell, this is the
        # type of attribute used in the article introducing the variational fair autoencoders
        ds.loc[:,8] = (ds.loc[:,8] == 'A92') | (ds.loc[:,8] == 'A95')

        ds.columns = self.all_columns()
        ds.to_csv(self.dataset_path(), index=False)
