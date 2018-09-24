import pandas
import sklearn.model_selection as ms
import sklearn.svm as svm
import numpy as np
import sys


df = pandas.read_csv(sys.argv[1])
y_columns = df.filter(regex=("y.*")).as_matrix()
s_columns = df.filter(regex=("s.*")).as_matrix()
h_columns = df.filter(regex=("h.*")).as_matrix()

h_train, h_test, s_train, s_test, y_train, y_test = ms.train_test_split(h_columns, s_columns, y_columns)

svc_y = svm.SVC()
svc_y.fit(h_train, y_train[:,1])
y_pred = svc_y.predict(h_test)

print("y accuracy: " + str(1.0 - np.mean(np.abs(y_test[:,1] - y_pred))))

svc_s = svm.SVC()
s_train = np.argmax(s_train, axis=1)
s_test = np.argmax(s_test, axis=1)
svc_s.fit(h_train, s_train)
s_pred = svc_s.predict(h_test)

print("s accuracy:" + str(1.0 - np.mean(np.abs(s_test - s_pred))))
