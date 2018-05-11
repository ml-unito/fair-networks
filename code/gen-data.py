import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression as LR

DS_SIZE = 10000
xmu = [0, 0]

xcorr = [[1,0.7],
        [0.7,1]]

zmu = [0, 0]
zcorr = [[1,0.4],
         [0.4,1]]

x = np.random.multivariate_normal(xmu,xcorr,DS_SIZE)

s = np.random.randint(5, size=DS_SIZE).reshape(10000,1)
z = np.array([np.random.multivariate_normal(np.multiply(zmu, si), zcorr) for si in s])

y = np.exp(15*x[:,0]/100) + -3*x[:,1]**5 + 100*s[:,0]**3 + 5*(z[:,0] - z[:,1])**6
y = y / np.max(np.abs(y))

ds = np.concatenate([x, s, z], axis=1)

trainx, trainy = ds[:8000,:], y[:8000]
testx, testy = ds[8000:, :], y[8000:]

transform = PolynomialFeatures(5)
trainx_ = transform.fit_transform(trainx)
testx_ = transform.fit_transform(testx)

dt = LR()
dt.fit(trainx_, trainy)
print(np.linalg.norm(dt.predict(testx_) - testy, 1)/len(testx_))
print(np.linalg.norm(dt.predict(trainx_) - trainy, 1)/len(trainx_))


predy = dt.predict(testx_)
srt = np.argsort(predy)

np.abs((predy[srt] - testy[srt]))[1999]


predy = dt.predict(trainx_)
plt.plot(predy, trainy, 'o')
plt.plot(predy, predy)
plt.axis('square')
plt.show()
