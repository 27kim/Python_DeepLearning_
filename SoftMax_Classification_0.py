# tensorflow import
import tensorflow as tf
import numpy as np

xy = np.loadtxt('./data/05train.txt', dtype='float32')
print(xy)
x_data = xy[:, 0: 3]
y_data = xy[:, 3:]
print(x_data.shape, y_data.shape)


# softmax 는 0으로 넣어도 된다고?
W = tf.Variable(tf.zeros([3, 3]))



# placeholder 지정
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

hypothesis = tf.nn.softmax(tf.matmul(X, W))

cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis=1))

train = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    sess.run(tf.global_variables_initializer())

    for step in range(50001):
        sess.run(train, feed_dict={X: x_data, Y: y_data})
        if step % 2000 == 0:
            print(step, sess.run(cost, feed_dict={X: x_data, Y: y_data}), sess.run(W))

    # 값을 넣어 예측하기. feed_dict 에 값을 넣어서 11시간 공부 7번 수업
    a = sess.run(hypothesis, feed_dict={X: [[1, 11, 7]]})
    # 예측값 출력
    print("예측값")
    print(a)
    print(a, sess.run(tf.argmax(a, 1)))

    c = sess.run(hypothesis, feed_dict={X: [[1, 1, 0]]})

    print(c)
    print(c, sess.run(tf.argmax(c, 1)))

    correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    print(sess.run(accuracy, feed_dict={X: x_data, Y: y_data}))
