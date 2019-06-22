import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from scipy.io import loadmat
from vfae_louizos.VFAE import VFAE
from itertools import combinations
from fair.datasets.adult_dataset import AdultDataset
from fair.datasets.bank_marketing_dataset import BankMarketingDataset
from fair.datasets.compas_dataset import CompasDataset
from fair.datasets.default_dataset import DefaultDataset
from fair.datasets.german_louizos_dataset import GermanLouizosDataset
import argparse

def discrimination(preds, corr_prob, stest, dim_s):
    disc, disc_p, tot_comb = 0, 0, 0
    for sval0, sval1 in combinations(range(dim_s), 2):
        y0, y0prob = preds[stest == sval0], corr_prob[stest == sval0]
        y1, y1prob = preds[stest == sval1], corr_prob[stest == sval1]
        disc += np.abs(np.mean(y0) - np.mean(y1))
        disc_p += np.abs(np.mean(y0prob) - np.mean(y1prob))
        tot_comb += 1
    disc /= tot_comb
    disc_p /= tot_comb
    return disc, disc_p


def accuracy_rf_lr(xtrain, ytrain, xtest, ytest):
    # random forest acc
    lr = RandomForestClassifier()
    lr.fit(xtrain, ytrain)
    preds = lr.predict(xtest)
    rf_a = (preds == ytest).sum() / (1. * ytest.shape[0])
    # lr acc
    lr = LogisticRegression()
    lr.fit(xtrain, ytrain)
    preds = lr.predict(xtest)
    lr_a = (preds == ytest).sum() / (1. * ytest.shape[0])
    return rf_a, lr_a


def random_chance(s):
    return max(np.bincount(s) / float(s.shape[0]))


def get_lr_pred_proba(xtrain, ytrain, xtest, ytest):
    lr = LogisticRegression()
    lr.fit(xtrain, ytrain)
    preds, pred_proba = lr.predict(xtest), lr.predict_proba(xtest)
    corr_prob = pred_proba[:, ytest]
    return preds, corr_prob

def old_main():
    prefix = 'packages/vfae_louizos/'
    # training set
    D = loadmat(prefix+'adult/train-1.mat')
    x, y, s = D['x'], D['y'].ravel(), D['s'].ravel().astype(np.int32)
    N, dim_x = x.shape
    dim_y, dim_s = np.unique(y).shape[0], np.unique(s).shape[0]
    # validation set
    Dv = loadmat(prefix+'adult/valid-1.mat')
    xv, yv, sv = Dv['x'], Dv['y'].ravel(), Dv['s'].ravel().astype(np.int32)
    # test set
    Dt = loadmat(prefix+'adult/test.mat')
    xt, yt, st = Dt['x'], Dt['y'].ravel(), Dt['s'].ravel().astype(np.int32)
    prior_y, batch_size = np.bincount(y) / float(y.shape[0]), 128
    print('y shape {}'.format(y.shape))
    print('s shape {}'.format(s.shape))
    print(y[0])

def reverse_onehot(dataset):
    x_train, y_train, s_train = dataset.train_all_data()
    x_val, y_val, s_val = dataset.val_all_data()
    x_test, y_test, s_test = dataset.test_all_data()
    y_train = np.argmax(y_train, axis=1)
    s_train = np.argmax(s_train, axis=1)
    y_test = np.argmax(y_test, axis=1)
    s_test = np.argmax(s_test, axis=1)
    y_val = np.argmax(y_val, axis=1)
    s_val = np.argmax(s_val, axis=1)
    return (x_train, y_train, s_train), (x_val, y_val, s_val), (x_test, y_test, s_test)


def main(args, use_MMD=True, use_s=True, random_seed=12345):
    dataset = args.dataset('./data')
    train_data, val_data, test_data = reverse_onehot(dataset)
    x_train, y_train, s_train = train_data
    x_val, y_val, s_val = val_data
    x_test, y_test, s_test = test_data
    N, dim_x = x_train.shape
    dim_y = np.unique(y_train).shape[0]
    dim_s = np.unique(s_train).shape[0]
    batch_size = 128
    prior_y = np.bincount(y_train) / float(y_train.shape[0])

    vfae = VFAE(N, dim_x, dim_s, dim_y, batch_size=batch_size, dim_h_en_z1=[200], dim_h_en_z2=[100], dim_h_de_z1=[100],
                dim_h_de_x=[200], dim_h_clf=[], dim_z1=50, dim_z2=50, L=1, iterations=1, nonlinearity='softplus',
                use_MMD=use_MMD, kernel_MMD='rbf_fourier', lambda_reg=20., supervised_rate=1., type_rec='binary',
                normalization='l2', regularization='l2', weight_decay=None, dropout_rate=0., learningRate=0.002,
                prior_y=prior_y, use_s=use_s, log_txt='adult.txt', random_seed=random_seed, optim_alg='adamax',
                beta1=0.9, beta2=0.999, polyak=True, beta3=0.9)

    _, _ = vfae.fit(x_train, s_train, y_train, xvalid=x_val, svalid=s_val, yvalid=y_val, verbose=False, print_every=10)

    # original s
    print('############### Original Representation ####################')
    brfs, blrs = accuracy_rf_lr(x_train, s_train, x_test, s_test)
    brfy, blry = accuracy_rf_lr(x_train, y_train, x_test, y_test)
    preds, corr_prob = get_lr_pred_proba(x_train, y_train, x_test, y_test)
    disc, disc_p = discrimination(preds, corr_prob, s_test, dim_s)
    rand_s, rand_y = random_chance(s_test), random_chance(y_test)
    print('RF accuracy on S:', brfs)
    print('LR accuracy on S:', blrs)
    print('RF accuracy on Y:', brfy)
    print('LR accuracy on Y:', blry)
    print('Baseline Discrimination:', disc)
    print('Baseline Discrimination proba:', disc_p)
    print('Random chance accuracy for S:', rand_s)
    print('Random chance accuracy for Y:', rand_y)
    print('###################### VFAE z1 ###############################')
    dummy_x, dummy_s = np.zeros((0, x_train.shape[1]), dtype=np.float32), np.zeros((0,), dtype=np.int32)
    invariant_z, invariant_z_test = vfae.transform(x_train, dummy_x, s_train, dummy_s), vfae.transform(x_test, dummy_x, s_test, dummy_s)
    brfs, blrs = accuracy_rf_lr(invariant_z, s_train, invariant_z_test, s_test)
    brfy, blry = accuracy_rf_lr(invariant_z, y_train, invariant_z_test, y_test)
    preds, corr_prob = get_lr_pred_proba(invariant_z, y_train, invariant_z_test, y_test)
    preds_vfae, prob_vfae = vfae.predict(x_test, dummy_x, s_test, dummy_s)
    disc, disc_p = discrimination(preds, corr_prob, s_test, dim_s)
    disc_vfae, disc_p_vfae = discrimination(preds_vfae, prob_vfae[:, y_test], s_test, dim_s)
    acc_y_vfae = (preds_vfae == y_test).sum() / (1. * y_test.shape[0])
    print( 'RF accuracy on S:', brfs)
    print( 'LR accuracy on S:', blrs)
    print( 'RF accuracy on Y:', brfy)
    print( 'LR accuracy on Y:', blry)
    print( 'Discrimination LR:', disc)
    print( 'Discrimination LR proba:', disc_p)
    print( 'Discrimination VFAE:', disc_vfae)
    print( 'Discrimination VFAE proba:', disc_p_vfae)
    print( 'VFAE accuracy on Y:', acc_y_vfae)


if __name__ == '__main__':
    # handle args here
    datasets = { 'adult': AdultDataset, 'bank': BankMarketingDataset, 'compas': CompasDataset, 'german': GermanLouizosDataset,
                  'default': DefaultDataset }
    parser = argparse.ArgumentParser(description='Train a VFAE.')
    parser.add_argument('--lr', metavar='lr', type=float, default=1e-4, help='The model learning rate')
    parser.add_argument('dataset', choices=['adult', 'bank', 'synth', 'default', 'german', 'compas'], help="dataset to be loaded")
    args = parser.parse_args()
    args.dataset = datasets[args.dataset]
    old_main()
    main(args)
