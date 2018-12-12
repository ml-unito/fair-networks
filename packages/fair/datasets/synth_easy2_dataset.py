from .synth_easy_dataset import SynthEasyDataset

class SynthEasy2Dataset(SynthEasyDataset):
    def dataset_path(self):
        return 'data/synth-easy2.csv'
