import tensorflow as tf

# 데이터
x_data = [1., 2., 3.]
y_data = [1., 2., 3.]

# y = wx + b
w = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.random_uniform([1], -1.0, 1.0))

hypothesis = w * x_data + b

# 식에 대입
cost = tf.reduce_mean(tf.square(hypothesis - y_data))
# Gradient Decent 가져오기
optimizer = tf.train.GradientDescentOptimizer(0.1)

train = optimizer.minimize(cost)

init = tf.global_variables_initializer()

# 세션 가져오기
sess = tf.Session()

sess.run(init)

for step in range(2001):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(cost), sess.run(w), sess.run(b))
