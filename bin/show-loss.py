from fair.datasets.synth_easy3_dataset import SynthEasy3Dataset
from fair.datasets.synth_easy2_dataset import SynthEasy2Dataset

from sklearn.svm import SVC
import numpy as np
import sys
import os
import pandas
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt

ds = SynthEasy2Dataset('data')

xs, ys, s = ds.train_all_data()
txs, tys, ts = ds.test_all_data()

rand = np.random.uniform(size=len(xs))
trand = np.random.uniform(size=len(txs))


def loss(xs, ys, s, txs, tys, ts, w):
    
    xs_ = (np.multiply(xs[:,0], w[0]) + np.multiply(rand, w[1])).reshape(-1,1)
    txs_ = (np.multiply(txs[:,0], w[0]) + np.multiply(trand, w[1])).reshape(-1,1)

    svc = SVC(probability=True)
    svc.fit(xs_, s[:,1])
    ts_ = svc.predict_proba(txs_)


    # SynthEasy3    
    # svc_y = SVC(probability=True)
    # svc_y.fit(np.hstack((xs_, xs[:,1].reshape(-1,1))), ys[:,1])
    # tys_ = svc_y.predict_proba(np.hstack((txs_, txs[:,1].reshape(-1,1))))

    # SynthEasy2
    svc_y = SVC(probability=True)
    svc_y.fit(xs_, ys[:,1])
    tys_ = svc_y.predict_proba(txs_)

    err = np.average(np.square(ts[:, 1] - ts_[:, 1]))
    m_loss = np.power(err - 0.5, 2)
    v_loss = -np.square(np.std(ts[:,1] - ts_[:,1]))
    y_loss = np.average(np.square(tys[:,1]-tys_[:,1]))

    return m_loss, v_loss, y_loss

def compute_losses():
    losses = []
    for w1 in np.linspace(-1, 1, num=20):
        for w2 in np.linspace(-1, 1, num=20):
            ms_l, vs_l, y_l = loss(xs, ys, s, txs, tys, ts, [w1, w2])
            losses.append([w1, w2, ms_l, vs_l, y_l])

    return np.array(losses)

def read_losses(path):
    df = pandas.read_csv(path)
    return df.as_matrix()


if os.path.exists("loss.csv"):
    losses = read_losses("loss.csv")
else:
    losses = compute_losses() 
    pandas.DataFrame(losses).to_csv("loss.csv", index=False)

print(len(losses))

fig1 = plt.figure()
fig2 = plt.figure()
fig3 = plt.figure()
fig4 = plt.figure()
fig5 = plt.figure()

ax = fig1.add_subplot(111, projection='3d')
bx = fig2.add_subplot(111, projection='3d')
cx = fig3.add_subplot(111, projection='3d')
dx = fig4.add_subplot(111, projection='3d')
ex = fig5.add_subplot(111, projection='3d')

ax.plot_trisurf(losses[:, 0], losses[:, 1], losses[:, 2])
bx.plot_trisurf(losses[:, 0], losses[:, 1], losses[:, 3])
cx.plot_trisurf(losses[:, 0], losses[:, 1], losses[:, 2] + losses[:, 3])
dx.plot_trisurf(losses[:, 0], losses[:, 1], losses[:, 4])
ex.plot_trisurf(losses[:, 0], losses[:, 1], losses[:, 2] + losses[:, 3] + losses[:, 4])

ax.set_xlabel('w1')
ax.set_ylabel('w2')
ax.set_zlabel('mean (s loss)')

bx.set_xlabel('w1')
bx.set_ylabel('w2')
bx.set_zlabel('var (s loss)')


cx.set_xlabel('w1')
cx.set_ylabel('w2')
cx.set_zlabel('mean - var (s loss)')

dx.set_xlabel('w1')
dx.set_ylabel('w2')
dx.set_zlabel('mean (y loss)')

ex.set_xlabel('w1')
ex.set_ylabel('w2')
ex.set_zlabel('combined loss')


total_loss = [losses[i, 2] + losses[i, 3] + 2*losses[i, 4] for i in range(len(losses))]
min_index = np.argmin(total_loss)
print("w1: {:6.2f} w2: {:6.2f}".format(losses[min_index, 0], losses[min_index, 1]))

w = [losses[min_index, 0], losses[min_index, 1]]
# w = [0,0]

xs_ = (np.multiply(xs[:, 0], w[0]) + np.multiply(rand, w[1])).reshape(-1, 1)
txs_ = (np.multiply(txs[:, 0], w[0]) + np.multiply(trand, w[1])).reshape(-1, 1)

svc = SVC()
svc.fit(xs_, s[:, 1])
ts_ = svc.predict(txs_)
print(np.average(ts_ != ts[:, 1]))

svc_y = SVC()
svc_y.fit(xs_, ys[:, 1])
tys_ = svc_y.predict(txs_)
print(np.average(tys_ != tys[:,1]))


  # SynthEasy3
  # svc_y = SVC(probability=True)
  # svc_y.fit(np.hstack((xs_, xs[:,1].reshape(-1,1))), ys[:,1])
  # tys_ = svc_y.predict_proba(np.hstack((txs_, txs[:,1].reshape(-1,1))))

  # SynthEasy2


# plt.show()
