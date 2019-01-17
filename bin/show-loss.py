from fair.datasets.synth_easy3_dataset import SynthEasy3Dataset
from fair.datasets.synth_easy2_dataset import SynthEasy2Dataset

from sklearn.svm import SVC
import numpy as np
import sys
import os
import pandas
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def loss(rand_avg, rand_var, xs, ys, s, txs, tys, ts, w):
    
    xs_ = (np.multiply(xs[:,0], w[0]) + np.multiply(rand, w[1])).reshape(-1,1)
    txs_ = (np.multiply(txs[:,0], w[0]) + np.multiply(trand, w[1])).reshape(-1,1)

    svc = SVC(probability=True)
    svc.fit(xs_, s[:,1])
    ts_ = svc.predict_proba(txs_)

    svc_y = SVC(probability=True)
    svc_y.fit(np.hstack((xs_, xs[:,1:])), ys[:,1])
    tys_ = svc_y.predict_proba(np.hstack((txs_, txs[:,1:])))

    err = np.average(np.square(ts[:, 1] - ts_[:, 1]))
    m_loss = np.power(err - rand_avg, 2)
    v_loss = np.square(np.var(ts[:,1] - ts_[:,1]) - rand_var)
    y_loss = np.average(np.square(tys[:,1]-tys_[:,1]))

    return m_loss, v_loss, y_loss


def compute_losses(rand_avg, rand_var):
    search_grid_size = 20
    losses = []
    for w1 in np.linspace(-1, 1, num=search_grid_size):
        for w2 in np.linspace(-1, 1, num=search_grid_size):
            ms_l, vs_l, y_l = loss(rand_avg, rand_var, xs,
                                   ys, s, txs, tys, ts, [w1, w2])
            losses.append([w1, w2, ms_l, vs_l, y_l])

    return np.array(losses)

def read_losses(path):
    df = pandas.read_csv(path)
    return df.as_matrix()

def print_loss(w, rand, trand):
    xs_ = (w[0]* xs[:, 0] + w[1] * rand).reshape(-1,1)
    txs_ = (w[0] * txs[:, 0] + w[1] * trand).reshape(-1, 1)

    xxs_ = np.hstack((xs_, xs[:, 1:]))
    txxs_ = np.hstack((txs_, txs[:, 1:]))

    svc = SVC()
    svc.fit(xxs_, s[:, 1])
    ts_ = svc.predict(txxs_)
    print("s accuracy: {:4.2f}".format(np.average(ts_ == ts[:, 1])))

    svc_y = SVC()
    svc_y.fit(xxs_, ys[:, 1])
    tys_ = svc_y.predict(txxs_)
    print("y accuracy: {:4.2f}".format(np.average(tys_ == tys[:, 1])))


def print_min_loss(losses, combined_loss, rand, trand):
    min_index = np.argmin(combined_loss)
    print("w1: {:6.2f} w2: {:6.2f}".format(losses[min_index, 0], losses[min_index, 1]))

    w = [losses[min_index, 0], losses[min_index, 1]]
    print_loss(w, rand, trand)




def plot_loss(losses, target_loss, label):
    fig1 = plt.figure()
    ax = fig1.add_subplot(111, projection='3d')
    ax.plot_trisurf(losses[:, 0], losses[:, 1],
                    target_loss, cmap=cm.get_cmap('autumn'))
    ax.set_xlabel('w1')
    ax.set_ylabel('w2')
    ax.set_zlabel(label)

def compute_random_performances(xs, s, txs, ts):
    svc = SVC()
    rand_xs = np.random.uniform(size=xs.shape)
    svc.fit(rand_xs, s[:,1])
    test_rand_data = np.random.uniform(size=txs.shape)
    ts_ = svc.predict(test_rand_data)
    misclassifications = ts[:,1] != ts_

    return np.average(misclassifications), np.var(misclassifications) 


ds = SynthEasy3Dataset('data')

xs, ys, s = ds.train_all_data()
txs, tys, ts = ds.test_all_data()

rand = np.random.uniform(size=len(xs))
trand = np.random.uniform(size=len(txs))

rand_avg, rand_std = compute_random_performances(xs, s, txs, ts)
print("Random model: mean s {:4.4} var s {:4.4}".format(rand_avg, rand_std))

if os.path.exists("loss.csv"):
    losses = read_losses("loss.csv")
else:
    losses = compute_losses(rand_avg, rand_std)
    pandas.DataFrame(losses).to_csv("loss.csv", index=False)


plot_loss(losses, losses[:, 2], 'mean (s loss)')
plot_loss(losses, losses[:, 3], 'var (s loss)')
plot_loss(losses, losses[:, 2] + losses[:,3], 'combined (s loss)')
plot_loss(losses, losses[:, 4], 'y loss')


fairness = 1
combined_loss = fairness*(losses[:, 2] + losses[:, 3]) + losses[:, 4]
plot_loss(losses, combined_loss, 'combined (all losses)')

print("\nMin loss")
print_min_loss(losses, combined_loss, rand, trand)

print("\nLoss of w=[0,0]")
print_loss([0.0,0,0], rand, trand)

plt.show()
