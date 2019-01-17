from sklearn.svm import SVC
import numpy as np

def estimate_mean_and_variance(num_features, num_examples=1000):
    svc = SVC()
    data = np.random.rand(num_examples, num_features)
    labels = [int(i > num_examples/2) for i in range(num_examples)]
    svc.fit(data, labels)
    preds = svc.predict(data)
    err = np.equal(labels, preds)
    mean = np.average(err)
    var = np.var(err)
    return mean, var