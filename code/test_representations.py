import pandas
import sklearn.model_selection as ms
import sklearn.svm as svm
import numpy as np
import sys
from sklearn.metrics import confusion_matrix

# file = "experiments/adult/adult-fair-networks-representations.csv"
file = sys.argv[1]

df = pandas.read_csv(file)
y_columns = df.filter(regex=("y.*")).values
s_columns = df.filter(regex=("s.*")).values
h_columns = df.filter(regex=("h.*")).values

h_train, h_test, s_train, s_test, y_train, y_test = ms.train_test_split(h_columns, s_columns, y_columns)

svc_y = svm.SVC()
svc_y.fit(h_train, y_train[:,1])
y_pred = svc_y.predict(h_test)

print("y accuracy: " + str(1.0 - np.mean(np.abs(y_test[:,1] - y_pred))))

confusion_matrix(y_pred, y_test[:,1])


svc_s = svm.SVC()
s_train = np.argmax(s_train, axis=1)
s_test = np.argmax(s_test, axis=1)
svc_s.fit(h_train, s_train)
s_pred = svc_s.predict(h_test)

print("s accuracy:" + str(1.0 - np.mean(np.abs(s_test - s_pred))))

print(confusion_matrix(s_test, s_pred))

sum(s_test == s_pred) / len(s_test)

sum(s_test == 0)
sum(s_test == 1)
sum(s_test == 2)
sum(s_test == 1) / float(len(s_test))
