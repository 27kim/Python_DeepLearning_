# tensorflow import
import tensorflow as tf

x_data = [
    [1., 1., 1., 1., 1.],
    [1., 0., 3., 0., 5.],
    [0., 2., 0., 4., 0.]]

y_data = [1, 2, 3, 4, 5]

W = tf.Variable(tf.random_uniform([1, 3], -1., 1.))
# b = tf.Variable(tf.random_uniform([1], -1., 1.))

hypothesis = tf.matmul(W, x_data)

cost = tf.reduce_mean(tf.square(hypothesis - y_data))

optimizer = tf.train.GradientDescentOptimizer(0.1)

train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(2001):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(cost), sess.run(W))
        # print(step, sess.run(cost), sess.run(W1), sess.run(W2), sess.run(b))
        #2000 4.8316907e-14 [[1.6412969e-07 9.9999994e-01 9.9999994e-01]]
