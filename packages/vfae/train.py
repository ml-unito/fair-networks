from __future__ import print_function

import matplotlib.pyplot as plt
from collections import OrderedDict

import os

import timeit
import pickle

import scipy.io as sio
import numpy as np
import theano
import theano.tensor as T
from sklearn.model_selection import train_test_split

import nnet as nn
import criteria	as er
import util
from vfae import VFAE, VFAE_struct, VFAE_coef, VFAE_training

import argparse

import sys
# workaround for importing stuff from parent dir. here the assumption is
# that you should be running this code from the main fair-networks directory.
sys.path.insert(0, 'packages')

try:
    from fair.datasets.bank_marketing_dataset import BankMarketingDataset
    from fair.datasets.adult_dataset import AdultDataset
    from fair.datasets.synth_dataset import SynthDataset
    from fair.datasets.german_louizos_dataset import GermanLouizosDataset
    from fair.datasets.default_dataset import DefaultDataset
    from fair.datasets.compas_dataset import CompasDataset                                                                                                                             
except ValueError as e:
    print('Please run this script from the main fair-networks folder, like so: python code/vfae/train.py [args]')
    print(e)
    sys.exit(1)
except ModuleNotFoundError as e:
    print('Please run this script from the main fair-networks folder, like so: python code/vfae/train.py [args]')
    print(e)
    sys.exit(1)


def main(args):
    dataset = datasets[args.dataset]('./data')
    train_xs, train_ys, train_s = dataset.train_all_data()

    struct = create_vfae_struct(args.lr, len(train_xs[0]))

    coef = VFAE_coef(
        alpha = 1000,
        beta = 100,
        chi = 1,
        D = 500,
        L = 1,
        optimize = 'Adam_update'
    )

    train_vfae(struct, coef, dataset)

def train_vfae(struct, coef, dataset):
    train_xs, train_ys, train_s = dataset.train_all_data()
    test_xs, test_ys, test_s = dataset.test_all_data()

    train_xs, val_xs, train_ys, val_ys = train_test_split(train_xs, train_ys, train_size=0.8)

    # looking at https://github.com/NCTUMLlab/Huang-Ching-Wei/blob/master/Experiments/Amazon_Reviews/DataPackage.py
    source_data = [(train_xs, train_ys), (val_xs, val_ys), (test_xs, test_ys)]
    target_data = [(train_xs, train_ys), (val_xs, val_ys), (test_xs, test_ys)]

    features_model, test_model, trained_param = VFAE_training(
        source_data=source_data,
        target_data=target_data,
        n_train_batches=20,
        n_epochs=50,
        struct = struct,
        coef = coef,
        description = 'prova'
    )


def create_vfae_struct(learning_rate, num_features):
    x_dim = num_features
    y_dim = 2
    d_dim = 2
    z_dim = 50                #dimension of latent feature
    a_dim = 50               #dimension of prior of latent feature
    h_zy_dim = 500             #dimension of hidden unit
    h_ay_dim = 100
    h_y_dim = 30
    activation = T.nnet.sigmoid

    struct = VFAE_struct()
    encoder_template = nn.NN_struct()


    struct.encoder1.share.layer_dim = [x_dim+d_dim, h_zy_dim]
    struct.encoder1.share.activation = [activation]
    struct.encoder1.share.learning_rate = [learning_rate, learning_rate]
    struct.encoder1.share.decay = [1, 1]

    struct.encoder1.mu.layer_dim = [h_zy_dim, z_dim]
    struct.encoder1.mu.activation = [None]
    struct.encoder1.mu.learning_rate = [learning_rate]
    struct.encoder1.mu.decay = [1, 1]
    struct.encoder1.sigma.layer_dim = [h_zy_dim, z_dim]
    struct.encoder1.sigma.activation = [None]
    struct.encoder1.sigma.learning_rate = [learning_rate, learning_rate]
    struct.encoder1.sigma.decay = [1, 1]

    struct.encoder2.share.layer_dim = [z_dim+y_dim, h_ay_dim]
    struct.encoder2.share.activation = [activation]
    struct.encoder2.share.learning_rate = [learning_rate, learning_rate]
    struct.encoder2.share.decay = [1, 1]
    struct.encoder2.mu.layer_dim = [h_ay_dim, a_dim]
    struct.encoder2.mu.activation = [None]
    struct.encoder2.mu.learning_rate = [learning_rate, learning_rate]
    struct.encoder2.mu.decay = [1, 1]
    struct.encoder2.sigma.layer_dim = [h_ay_dim, a_dim]
    struct.encoder2.sigma.activation = [None]
    struct.encoder2.sigma.learning_rate = [learning_rate, learning_rate]
    struct.encoder2.sigma.decay = [1, 1]

    struct.encoder3.layer_dim = [z_dim, y_dim]
    struct.encoder3.activation = [T.nnet.softmax]
    struct.encoder3.learning_rate = [learning_rate, learning_rate]
    struct.encoder3.decay = [1, 1]

    struct.decoder1.share.layer_dim = [z_dim+d_dim, h_zy_dim]
    struct.decoder1.share.activation = [activation]
    struct.decoder1.share.learning_rate = [learning_rate, learning_rate]
    struct.decoder1.share.decay = [1, 1]
    struct.decoder1.mu.layer_dim = [h_zy_dim, x_dim]
    struct.decoder1.mu.activation = [None]
    struct.decoder1.mu.learning_rate = [learning_rate, learning_rate]
    struct.decoder1.mu.decay = [1, 1]
    struct.decoder1.sigma.layer_dim = [h_zy_dim, x_dim]
    struct.decoder1.sigma.activation = [None]
    struct.decoder1.sigma.learning_rate = [learning_rate, learning_rate]
    struct.decoder1.sigma.decay = [1, 1]

    struct.decoder2.share.layer_dim = [a_dim+y_dim, h_ay_dim]
    struct.decoder2.share.activation = [activation]
    struct.decoder2.share.learning_rate = [learning_rate, learning_rate]
    struct.decoder2.share.decay = [1, 1]
    struct.decoder2.mu.layer_dim = [h_ay_dim, z_dim]
    struct.decoder2.mu.activation = [None]
    struct.decoder2.mu.learning_rate = [learning_rate, learning_rate]
    struct.decoder2.mu.decay = [1, 1]
    struct.decoder2.sigma.layer_dim = [h_ay_dim, z_dim]
    struct.decoder2.sigma.activation = [None]
    struct.decoder2.sigma.learning_rate = [learning_rate, learning_rate]
    struct.decoder2.sigma.decay = [1, 1]

    return struct

if __name__ == '__main__':
    # handle args here
    datasets = { 'adult': AdultDataset, 'bank': BankMarketingDataset, 'synth': SynthDataset }
    parser = argparse.ArgumentParser(description='Train a VFAE.')
    parser.add_argument('--lr', metavar='lr', type=float, default=1e-4, help='The model learning rate')
    parser.add_argument('dataset', choices=['adult', 'bank', 'synth', 'default', 'german-louizos', 'compas'], help="dataset to be loaded")
    args = parser.parse_args()
    main(args)
