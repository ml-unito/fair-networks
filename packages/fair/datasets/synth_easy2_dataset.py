from .synth_easy_dataset import SynthEasyDataset

class SynthEasy2Dataset(SynthEasyDataset):
    def name(self):
        return "SynthEasy2"

    def dataset_path(self):
        return 'data/synth-easy2.csv'
