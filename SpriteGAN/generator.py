import numpy as np
import tensorflow as tf
from utils import batch_normalization

# # TEMP:
zdim = 100
ydim = 1 # number of classes, we've only got two but should work with this anyway


class Generator(object):
    """docstring for Generator."""
    def __init__(self, stddev=0.02):
        self.stddev = stddev

    def __call__(self, zs, ys, is_training):
        batch_dim = tf.shape(zs)[0]
        with tf.variable_scope('generator', initializer=tf.truncated_normal_initializer(stddev=self.stddev)):
            inputs = zs
            with tf.variable_scope('volume'): # Project and reshape
                W_p = tf.get_variable('W', [zdim, 4 * 4 * 16])
                z_p = tf.matmul(inputs, W_p)
                bn_p = batch_normalization(z_p, center=False,
                                            scale=False,
                                            training=is_training)
                a_p = tf.nn.relu(bn_p)
                reshaped_a_p = tf.reshape(a_p, [-1, 4, 4, 16]) # 4x4

            with tf.variable_scope('deconv1'): # deconvolution here means conv transpose
                filters_1 = tf.get_variable('filters', [2, 2, 32, 16])    # output_size = strides * (input_size-1) + kernel_size - 2*padding

                                                                          #  strides, input_size, kernel_size, padding are integer padding is zero for 'valid'
                deconv_1 = tf.nn.conv2d_transpose(reshaped_a_p, filters_1,
                                                    [batch_dim] + [8, 8, 32],
                                                    [1, 2, 2, 1], padding='VALID')
                bn_1 = batch_normalization(tf.reshape(deconv_1, [batch_dim] + [8, 8, 32]),
                                            center=False,
                                            scale=False,
                                            training=is_training)
                a_1 = tf.nn.relu(bn_1) # 8x8

            with tf.variable_scope('deconv2'): # deconvolution here means conv transpose
                filters_2 = tf.get_variable('filters', [2, 2, 64, 32])
                deconv_2 = tf.nn.conv2d_transpose(a_1, filters_2,
                                                    [batch_dim] + [16, 16, 64],
                                                    [1, 2, 2, 1], padding='VALID')
                bn_2 = batch_normalization(tf.reshape(deconv_2, [batch_dim] + [16, 16, 64]),
                                            center=False,
                                            scale=False,
                                            training=is_training)
                a_2 = tf.nn.relu(bn_2)

            with tf.variable_scope('deconv3'): # deconvolution here means conv transpose
                filters_3 = tf.get_variable('filters', [2, 2, 3, 64])
                deconv_3 = tf.nn.conv2d_transpose(a_2, filters_3, [batch_dim] + [32, 32, 3],
                                                    [1, 2, 2, 1], padding='VALID')
                # bn_3 = batch_normalization(tf.reshape(deconv_3, [batch_dim] + v['output']),
                #                             center=v['bn']['center'],
                #                             scale=v['bn']['scale'],
                #                             training=is_training) if 'bn' in v else deconv_3
                a_3 = tf.nn.tanh(deconv_3)

        return a_3