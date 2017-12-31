import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

def add_layer(input, in_size, out_size, activation_func=None):
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)                # not zero
    Wx_plus_b = tf.matmul(input, Weights) + biases
    if activation_func is None:
        output = Wx_plus_b
    else:
        output = activation_func(Wx_plus_b)
    return output

def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs:v_xs})
    correct_pre = tf.equal(tf.arg_max(y_pre, 1), tf.arg_max(v_ys, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pre, tf.float32))
    res = sess.run(accuracy, feed_dict={xs:v_xs, ys:v_ys})
    return res

# def placeholder
xs = tf.placeholder(tf.float32, [None, 784])   # 28 * 28
ys = tf.placeholder(tf.float32, [None, 10])


# in_size is the feature size
prediction = add_layer(xs, 784, 10, activation_func=tf.nn.softmax)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction), reduction_indices=[1]))



loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),
                                    reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
init = tf.global_variables_initializer()


with tf.Session() as sess:
    sess.run(init)
    for i in range(10000):
        batch_xs, batch_ys = mnist.train.next_batch(100)

        sess.run(train_step, feed_dict={xs:batch_xs, ys:batch_ys})
        if i % 50 == 0:
            print(compute_accuracy(mnist.test.images, mnist.test.labels))
