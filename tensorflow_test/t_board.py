import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


## tensorboard --logdir='logs/'
def add_layer(input, in_size, out_size, n_layer, activation_func=None):
    layer_name = "layer%s" % n_layer
    with tf.name_scope('layer'):
        with tf.name_scope('weights'):
            Weights = tf.Variable(tf.random_normal([in_size, out_size]), name='W')
            tf.summary.histogram(layer_name + "/weight", Weights)
        with tf.name_scope('bias'):
            biases = tf.Variable(tf.zeros([1, out_size]) + 0.1, name='b')                # not zero
            tf.summary.histogram(layer_name + "/biases", biases)
        with tf.name_scope('Wx_b'):
            Wx_plus_b = tf.matmul(input, Weights) + biases
    if activation_func is None:
        output = Wx_plus_b
    else:
        output = activation_func(Wx_plus_b)
    tf.summary.histogram(layer_name + "/output", output)
    return output

x_data = np.linspace(-1, 1, 300)[:, np.newaxis]    # 300 x 1
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data) - 0.5 + noise

with tf.name_scope('input'):
    xs = tf.placeholder(tf.float32, [None, 1], name="x_input")
    ys = tf.placeholder(tf.float32, [None, 1], name="y_input")


# in_size is the feature size
l1 = add_layer(xs, 1, 10, n_layer=1, activation_func=tf.nn.relu)
prediction = add_layer(l1, 10, 1, n_layer=2, activation_func=None)

with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),
                                reduction_indices=[1]))
    tf.summary.scalar('loss', loss)
with tf.name_scope('train_step'):
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
init = tf.global_variables_initializer()

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.scatter(x_data, y_data)
plt.ion()
plt.show()

with tf.Session() as sess:
    sess.run(init)
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter("logs/", sess.graph)
    for i in range(1000):
        sess.run(train_step, feed_dict={xs:x_data, ys:y_data})
        if i % 50 == 0:
            print(sess.run(loss, feed_dict={xs:x_data, ys:y_data}))
            res = sess.run(merged, feed_dict={xs:x_data, ys:y_data})
            writer.add_summary(res, i)