from .dataset_base import DatasetBase
import pandas
import os.path
import logging

class GermanLouizosDataset(DatasetBase):
    DOWNLOAD_PATH = 'data/german-full-louizos.csv'

    """
    Helper class allowing to download and load into memory the german dataset as given by Louizos et al.
    """

    def name(self):
        return "German-Louizos"

    def all_columns(self):
        ext = list(range(60))
        attr_names = ['x_{}'.format(e) for e in ext]
        attr_names.extend(['s', 'y'])
        return attr_names

    def one_hot_columns(self):
        return [ "s", "y" ]

    def dataset_path(self):
        return "%s/german-full-louizos.csv" % (self.workingdir)

    def sep(self):
        return ','

    def files_to_retrieve(self):
        return [
            ('https://datacloud.di.unito.it/index.php/s/dxAESXoqoBooNpG/download',
             self.DOWNLOAD_PATH),
        ]

    def sensible_columns(self):
        return["s"]

    def y_columns(self):
        return ["y"]

    def prepare_all(self):
        if os.path.isfile(self.dataset_path()):
            logging.info("German-louizos dataset already exist. Using existing version.")
            return

        ds = pandas.read_csv(GermanLouizosDataset.DOWNLOAD_PATH, sep=',')

        ds.columns = self.all_columns()
        ds.to_csv(self.dataset_path(), index=False)
