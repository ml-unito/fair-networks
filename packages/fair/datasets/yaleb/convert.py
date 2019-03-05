import numpy
import pandas as pd
import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler

def load_pickle(path):
    f = open(path, 'r')
    data_dict = pickle.load(f)
    x = pd.DataFrame(data_dict['x'])
    y = pd.DataFrame(data_dict['t'])
    s = pd.DataFrame(data_dict['light'])
    y = y.rename({0: 'y'}, axis=1)
    s = s.rename({0: 's'}, axis=1)
    ext = range(0, 504)
    x = x.rename({e: 'x_' + str(e) for e in ext}, axis=1)
    return x, y, s


files = ['set_{}.pdata'.format(i) for i in range(0,5)]
test_file = 'test.pdata'
x = pd.DataFrame([])
y = pd.DataFrame([])
s = pd.DataFrame([])

for path in files:
    tx, ty, ts = load_pickle(path)
    x = pd.concat([x, tx], axis=0)
    y = pd.concat([y, ty], axis=0)
    s = pd.concat([s, ts], axis=0)

y = pd.DataFrame(y)
s = pd.DataFrame(s)
train = pd.concat([x, y, s], axis=1)

xt, yt, st = load_pickle(test_file)
test = pd.concat([xt, yt, st], axis=1)
test = test[test['s'] < 5]

scaler = MinMaxScaler()
#x = scaler.fit_transform(train.iloc[:, :504])
#xt = scaler.transform(test.iloc[:, :504])
x = train.iloc[:, :504]
xt = test.iloc[:, :504]

c = LogisticRegression()
c.fit(x, train.iloc[:, 504])
y_pred = c.predict(xt)


print('Accuracy check:')
print(sum(np.equal(test.iloc[:, 504], y_pred)) / float(len(y_pred)))

train.to_csv('yale_train.csv', index=False)
test.to_csv('yale_test.csv', index=False)

