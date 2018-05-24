from dataset_base import DatasetBase

class SynthDataset(DatasetBase):
    def all_columns(self):
        return ["x1","x2","s","z1","z2","y"]

    def one_hot_columns(self):
        return ["s", "y"]

    def dataset_path(self):
        return 'data/synth-full.csv'

    def sep(self):
        return ','

    def files_to_retrieve(self):
        return []

    def prepare_all(self):
        pass

    def num_y_columns(self):
        return 2
