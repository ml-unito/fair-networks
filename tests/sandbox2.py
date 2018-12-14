import pandas
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression


from fair.datasets.synth_easy2_dataset import SynthEasy2Dataset

#
# original data
#
df = pandas.read_csv("data/synth-easy2.csv")

x = df["x"].as_matrix()
s = df["s"].as_matrix()

test_x = x[:100]
test_s = s[:100]

x = x[100:]
s = s[100:]


scaler = MinMaxScaler()
xx = np.reshape(x, [len(x), 1])
test_xx = np.reshape(test_x, [len(test_x), 1])

# HERE IS THE PROBLEM: scaling the data significantly
# decrease the prediction performances on s
# 
xxx = scaler.fit_transform(xx.astype(np.float64))
test_xxx = scaler.fit_transform(test_xx.astype(np.float64))


svc0 = SVC()
svc0.fit(xx, s)
acc0 = sum(svc0.predict(xx) == s)/len(s)

svc = SVC()
svc.fit(xxx, s)
acc1 = sum(svc.predict(xxx) == s)/len(s)


assert abs(acc1-acc0) < 0.01, "predictions on scaled and unscaled data should be the same (they are instead: {} and {})".format(acc0, acc1)

# But it is a problem of svms!

tree0 = DecisionTreeClassifier()
tree0.fit(xx,s)
tacc0 = sum(tree0.predict(xx) == s)/len(s)

tree1 = DecisionTreeClassifier()
tree1.fit(xxx, s)
tacc1 = sum(tree1.predict(xxx) == s)/len(s)

assert abs(tacc1-tacc0) < 0.01, "predictions on scaled and unscaled data should be the same (they are instead: {} and {})".format(tacc0, tacc1)

# Le'ts try logistic regression

log0 = LogisticRegression()
log0.fit(xx, s)
lacc0 = sum(log0.predict(xx) == s)/len(s)


#
#  Working Dataset
#

ds = SynthEasy2Dataset('data')

x2,y2,s2 = ds.train_all_data()
ss2 = np.argmax(s2, axis=1)

svc2 = SVC()
svc2.fit(x2,ss2)

acc2 = sum(svc2.predict(x2) == ss2)/len(ss2)

#
# Original representation 
# 

dor = pandas.read_csv("experiments/synth_easy2_2/representations/original_repr.csv")

x3 = dor["h_0"].as_matrix()
xx3 = np.reshape(x3, [len(x3), 1])
s3 = dor[["s_0","s_1"]].as_matrix()
ss3 = np.argmax(s3, axis=1)

sv3 = SVC()
sv3.fit(xx3, ss3)

acc3 = sum(sv3.predict(xx3) == ss3)/len(ss3)

#
# They should be all roughly the same...
#

assert abs(acc1 - acc2) < 0.001, "predictions on original csv and on the dataset should be same (they are instead: {} and {})".format(acc1, acc2)
assert abs(acc2 - acc3) < 0.001, "predictions on the dataset and on original_repr should be same (they are instead: {} and {})".format(acc2, acc3)
assert abs(acc1 - acc2) < 0.001, "predictions on the original csv and on original_repr should be same (they are instead: {} and {})".format(acc1, acc3)
