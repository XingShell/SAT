
import numpy as np
import tensorflow as tf


def conv2d(x, channel, k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02, padding='VALID', name='conv2d'):
    with tf.variable_scope(name):
        w = tf.get_variable('weights', [k_h, k_w, x.get_shape()[-1], channel],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        biases = tf.get_variable('biases', shape=[channel], initializer=tf.zeros_initializer())
    conv = tf.nn.conv2d(x, w, strides=[1, d_h, d_w, 1], padding=padding)
    conv = tf.nn.bias_add(conv, biases)
    return conv
def get_conv_lens(lengths):
    return tf.floor_div(lengths - 1, 2)

x = tf.placeholder(shape=(3,None,None,1), dtype=tf.float32)
lens = tf.placeholder(shape=(3), dtype=tf.int32)

'''Masking'''
conv_lens = get_conv_lens(lens)
mask = tf.sequence_mask(conv_lens)
mask = tf.expand_dims(tf.expand_dims(mask, -1), -1)
mask = tf.to_float(mask)

y = conv2d(x, channel=1, k_w=2, k_h=2, d_w=2, d_h=2, name='conv')
y = tf.multiply(y, mask)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    feed_dict = {
        x: np.random.normal(size=(3, 9, 9, 1)),
        lens: [5, 7, 9]
    }
    out, l = sess.run([y, conv_lens], feed_dict=feed_dict)

print('conv_output_lens:', l)
for i in range(3):
    print("Sample {}, len:{}".format(i, l[i]))
    print(out[i, :, :, 0])
