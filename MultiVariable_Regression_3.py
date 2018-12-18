# tensorflow import
import tensorflow as tf
import numpy as np


xy = np.loadtxt('./data/test-score.csv', delimiter=',', dtype='float32')
print(xy)
x_data = xy[:, 0: -1]
y_data = xy[:, [-1]]
print(x_data.shape, y_data.shape)


W = tf.Variable(tf.random_uniform([3, 1], -1., 1.))

hypothesis = tf.matmul( x_data, W)

cost = tf.reduce_mean(tf.square(hypothesis - y_data))

optimizer = tf.train.GradientDescentOptimizer(0.0000000001)

train = optimizer.minimize(cost)

sess = tf.Session()

sess.run(tf.global_variables_initializer())

for step in range(2001):
    sess.run(train)
    if step % 20 == 0 :
        print(step, sess.run(cost), sess.run(W))
