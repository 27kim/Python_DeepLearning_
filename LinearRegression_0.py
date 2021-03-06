# tensorflow import
import tensorflow as tf

# 데이터
x_data = [1., 2., 3.]
y_data = [1., 2., 3.]

# y = wx + b
w = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.random_uniform([1], -1.0, 1.0))

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

hypothesis = w * X + b

# cost 구하는 식 생성
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# Gradient Decent 가져오기 (2차 함수이므로 결국에는 기울기가 0인 곳이 최소값일 것)
optimizer = tf.train.GradientDescentOptimizer(0.1)

# optimizer 가 최소값 구할 수 있도록
train = optimizer.minimize(cost)

# tensorflow 사용하기 위한 변수 초기화
init = tf.global_variables_initializer()

# 세션 가져오기
sess = tf.Session()

# variable 을 초기화
sess.run(init)

for step in range(2001):
    sess.run(train, feed_dict={X: x_data, Y: y_data})
    if step % 20 == 0:
        print(step, sess.run(cost, feed_dict={X: x_data, Y: y_data}), sess.run(w), sess.run(b))

print("결과 ? %s" %sess.run(hypothesis, feed_dict={X: 5}))
