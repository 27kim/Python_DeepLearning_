# tensorflow import
import tensorflow as tf

# x1_data = [1., 0., 3., 0., 5.]
# x2_data = [0., 2., 0., 4., 0.]
# x_data = [x1_data, x2_data]

x_data = [[1., 0., 3., 0., 5.],
          [0., 2., 0., 4., 0.]]

y_data = [1, 2, 3, 4, 5]

# W1 = tf.Variable(tf.random_uniform([1], -1., 1.))
# W2 = tf.Variable(tf.random_uniform([1], -1., 1.))
W = tf.Variable(tf.random_uniform([1, 2], -1., 1.))
b = tf.Variable(tf.random_uniform([1], -1., 1.))

# hypothesis = W1 * x1_data + W2 * x2_data + b
hypothesis = tf.matmul(W, x_data) + b

cost = tf.reduce_mean(tf.square(hypothesis - y_data))

optimizer = tf.train.GradientDescentOptimizer(0.1)

train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(2001):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(cost), sess.run(W), sess.run(b))
        # print(step, sess.run(cost), sess.run(W1), sess.run(W2), sess.run(b))