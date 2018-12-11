import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix


from fair.datasets.synth_easy_dataset import SynthEasyDataset


class Model:
    def __init__(self, num_features, num_y):
        self.x = tf.placeholder(tf.float32, shape=[None, num_features])
        self.y = tf.placeholder(tf.float32, shape=[None, num_y])

        self.in_layer = self.x
        self.h1_layer = tf.layers.dense(
            self.in_layer, 10, kernel_initializer=tf.truncated_normal_initializer())
        self.out = tf.layers.dense(
            self.h1_layer, num_y, kernel_initializer=tf.truncated_normal_initializer())

        self.loss = tf.losses.sigmoid_cross_entropy(self.y, self.out)

        equal_check = tf.equal(
            tf.argmax(self.y, axis=1), tf.argmax(self.out, axis=1))
        self.accuracy = tf.reduce_mean(tf.cast(equal_check, tf.float32))


EPOCHS=50
BATCH_SIZE=200

ds = SynthEasyDataset("data")

xs,ys,s = ds.train_all_data()
test_xs, test_ys, test_s = ds.test_all_data()

num_features = xs.shape[1]
num_ys = ys.shape[1]

model = Model(num_features, num_ys)

session = tf.Session()
optimizer = tf.train.AdamOptimizer()
train_step = optimizer.minimize(model.loss)

init = tf.global_variables_initializer()
session.run(init)

for epoch in range(EPOCHS):
    batch_pos = 0
    while batch_pos < len(xs):
        batch_x = xs[batch_pos:(batch_pos+BATCH_SIZE)]
        batch_y = ys[batch_pos:(batch_pos+BATCH_SIZE)]
        batch_pos += BATCH_SIZE

        session.run(train_step, feed_dict={model.x: batch_x, model.y: batch_y})

    print("loss:" + str(session.run(model.loss,
                                    feed_dict={model.x: xs, model.y: ys})))
    print("train:" + str(session.run(model.accuracy,
                                     feed_dict={model.x: xs, model.y: ys})))


random_preds = np.reshape(np.random.uniform(size=len(test_ys)),[len(test_ys), 1])
random_preds = np.hstack([random_preds, 1-random_preds])

print("Random accuracy:{:2.4f}".format(np.average(np.equal( np.argmax(test_ys, axis=1), np.argmax(random_preds, axis=1) ))))
print(confusion_matrix(np.argmax(test_ys, axis=1), np.argmax(random_preds, axis=1)))

majority_class = np.argmax(ys.sum(axis=0))
print("Majority class accuracy:{:2.4f}".format(np.average(np.equal(majority_class, np.argmax(test_ys, axis=1)))))
print(confusion_matrix(np.argmax(test_ys, axis=1), np.repeat(majority_class, len(test_ys))))


print("Network accuracy:{:2.4f}".format(session.run(model.accuracy, feed_dict={model.x: test_xs, model.y: test_ys})))
network_ys = session.run(model.out, feed_dict={
                         model.x: test_xs, model.y: test_ys})
cnf = confusion_matrix(np.argmax(test_ys, axis=1), np.argmax(network_ys, axis=1))
print(cnf)
print("cnf accuracy:{:2.4f}".format((cnf[0,0]+cnf[1,1])/cnf.sum()))

# Network accuracy: 0.8972
# [[3867  101]
#  [364  190]]
