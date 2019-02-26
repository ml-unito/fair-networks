import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVC


def first_dataset():
    DS_SIZE = 10000
    xmu = [1, 1]

    xcorr = np.array([[1,0.7],
                    [0.7,1]])

    zmu = [1, 1]
    zcorr = [[1,0.4],
            [0.4,1]]

    x = np.random.multivariate_normal(xmu,xcorr,DS_SIZE)

    s = np.random.randint(5, size=DS_SIZE)
    zmus = [np.multiply(zmu, si) for si in s]
    z = np.array([np.random.multivariate_normal(mu, zcorr) for mu in zmus])

    # y = np.exp(15*np.sin(x[:,0])) + -3*x[:,1]**5 + 100*s[:,0]**3 + 5*(z[:,0] - z[:,1])**6
    # y = y / np.max(np.abs(y)) > 0.001



    y =  [np.random.multivariate_normal(xi * (si+1), xcorr * (si+1)) for xi,si in zip(x,s)]
    y = np.array([np.exp(x1 + x2) for x1, x2 in y])
    y = y > 368.5

    sum(y)

    # ds = np.concatenate([x, s.reshape(10000,1), z], axis=1)
    # ds = np.concatenate([x,s.reshape(10000,1)], axis=1)
    # ds = np.concatenate([x,z],axis=1)
    # ds = x
    # ds = np.concatenate([s.reshape(10000,1),z], axis=1)
    # ds = s.reshape(10000,1)
    ds = z

    df = pd.DataFrame(np.concatenate([ds, np.reshape(y, [10000,1])], axis=1) , columns=["x1", "x2", "s", "z1", "z2", "y"])
    # df.to_csv("data/synth-full.csv", sep=",", index=False)

    #
    #

    yy = y
    svc = SVC()
    model = svc.fit(ds, yy)
    sum((model.predict(ds) == yy) == True) / len(ds)
    #
    #plt.scatter(x[:,1], z[:,1], c=y, marker='.')


    # df = pd.DataFrame(np.concatenate([testx, np.reshape(testy, [2000,1])], axis=1) , columns=["x1", "x2", "s", "z1", "z2", "y"])
    # df.to_csv("synth-test.csv", sep=",")


    # transform = PolynomialFeatures(3)
    # trainx_ = transform.fit_transform(trainx)
    # testx_ = transform.fit_transform(testx)
    #
    # dt = LR()
    # dt.fit(trainx_, trainy)
    # print(np.linalg.norm(dt.predict(testx_) - testy, 1)/len(testx_))
    # print(np.linalg.norm(dt.predict(trainx_) - trainy, 1)/len(trainx_))
    #
    #
    # predy = dt.predict(testx_)
    # srt = np.argsort(predy)
    #
    # np.abs((predy[srt] - testy[srt]))[1999]
    #
    #
    # predy = dt.predict(trainx_)
    # plt.plot(predy, trainy, 'o')
    # plt.plot(predy, predy)
    # plt.axis('square')
    # plt.show()


def synth_easy():
    DS_SIZE = 1000
    x = np.reshape(np.random.randint(0, high=50001, size=DS_SIZE), [DS_SIZE,1])
    s = np.reshape(np.random.randint(0, high=2, size=DS_SIZE), [DS_SIZE,1])
    noise = np.reshape(np.random.normal(scale=2000, size=DS_SIZE), [DS_SIZE, 1])
    y = (x + s*10000 + noise > 30000).astype(int)


    data = np.hstack([x,s,y])
    pd.DataFrame(data, columns=["x", "s", "y"]).to_csv("data/synth-easy.csv", index=False)


def synth_easy2():
    DS_SIZE = 1000

    x = np.reshape(np.random.randint(0, high=50001, size=DS_SIZE), [DS_SIZE, 1])
    s = np.reshape(np.random.randint(0, high=2, size=DS_SIZE), [DS_SIZE, 1])

    x = x + s*30000
    noise = 0
    # noise = np.reshape(np.random.normal(scale=2000, size=DS_SIZE), [DS_SIZE, 1])

    y = (x + noise > 30000).astype(int)

    data = np.hstack([x, s, y])
    pd.DataFrame(data, columns=["x", "s", "y"]).to_csv(
        "data/synth-easy2.csv", index=False)

def synth_easy3():
    DS_SIZE = 1000

    x2 = np.reshape(np.random.randint(0, high=50001, size=DS_SIZE), [DS_SIZE, 1])
    s = np.reshape(np.random.randint(0, high=2, size=DS_SIZE), [DS_SIZE, 1])

    noise = 0
    # noise = np.reshape(np.random.normal(scale=2000, size=DS_SIZE), [DS_SIZE, 1])

    x1 = s * 20000

    y = (x1 + x2 + noise > 35000).astype(int)

    data = np.hstack([x1, x2, s, y])
    pd.DataFrame(data, columns=["x1", "x2", "s", "y"]).to_csv(
        "data/synth-easy3.csv", index=False)

def synth_easy4():
    DS_SIZE = 1000

    x2 = np.reshape(np.random.randint(0, high=50001, size=DS_SIZE), [DS_SIZE, 1])
    s = np.reshape(np.random.randint(0, high=2, size=DS_SIZE), [DS_SIZE, 1])

    noise = 0
    # noise = np.reshape(np.random.normal(scale=2000, size=DS_SIZE), [DS_SIZE, 1])

    x1 = s * 20000

    y = (x1 + x2 + noise > 35000).astype(int)

    data = np.hstack([x1, x2, s, s, y])
    pd.DataFrame(data, columns=["x1", "x2", "hidden_s", "s", "y"]).to_csv(
        "data/synth-easy3.csv", index=False)



synth_easy4()

