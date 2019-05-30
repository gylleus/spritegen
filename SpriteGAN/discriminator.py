import numpy as np
import tensorflow as tf
from utils import get_shape, batch_normalization

ydim=1
zdim=100

def lkrelu(x, slope=0.01):
    return tf.maximum(slope * x, x)

class Discriminator(object):
    def __init__(self, stddev=0.02):
        self.stddev = stddev

    def __call__(self, xs, ys, is_training, reuse=None):
        batch_dim = tf.shape(xs)[0]
        with tf.variable_scope('discriminator', initializer=tf.truncated_normal_initializer(stddev=self.stddev), reuse=reuse):
            with tf.variable_scope('conv1'):
                filters_1 = tf.get_variable('filters', [4, 4, 3, 32])
                conv_1 = tf.nn.conv2d(xs, filters_1, [1, 2, 2, 1], padding='VALID')

                # The messy line below adds y as a channel to conv_1 as described in ICGAN paper
                #conv_1_concat_ys = tf.concat([conv_1, tf.tile(tf.reshape(ys, [-1, 1, 1, get_shape(ys)[-1]]),
                                                                #[1, tf.shape(conv_1)[1], tf.shape(conv_1)[2], 1])], axis=3)
                a_1 = lkrelu(conv_1, slope=0.2)#'#conv_1_concat_ys, slope=0.2)

            with tf.variable_scope('conv2'):
                filters_2 = tf.get_variable('filters', [4, 4, 32, 16])
                conv_2 = tf.nn.conv2d(a_1, filters_2, [1, 2, 2, 1], padding='VALID')
                bn_2 = batch_normalization(conv_2,  center=False,
                            scale=False, training=is_training)
                a_2 = lkrelu(bn_2, slope=0.2)

            with tf.variable_scope('conv3'):
                filters_3 = tf.get_variable('filters', [4, 4, 16, 8])
                conv_3 = tf.nn.conv2d(a_2, filters_3, [1, 2, 2, 1], padding='VALID')
                bn_3 = batch_normalization(conv_3,  center=False,
                                            scale=False, training=is_training)
                a_3 = lkrelu(bn_3, slope=0.2)

            with tf.variable_scope('output'):
                W_o = tf.get_variable('W', [128, 1])
                z_o = tf.matmul(tf.reshape(a_3, [-1] + [4 * 4 * 8, 1][:1]), W_o)
                a_o = tf.nn.sigmoid(z_o)
        return a_o