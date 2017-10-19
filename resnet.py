# -*- coding: utf-8 -*-
# Date    : 2017-09-05 19:21:30
# Author  : Ssj
# Copyright 2017 Shijie. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ------------------------------------------------------------------------

import os
import numpy as np
import tensorflow as tf




def _variable_on_cpu(name, shape, initializer):
    with tf.device('/cpu:0'):
        var = tf.get_variable(
            name, shape, initializer=initializer, dtype=tf.float32)
    return var


def _variable_with_weight_decay(name, shape, wd=0.0001):
    var = _variable_on_cpu(name, shape, tf.contrib.layers.xavier_initializer())
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weights_loss')
        tf.add_to_collection('losses', weight_decay)
    return var


def conv_layer(name, x, kernel_shape, strides, wd=0.0001):
    with tf.variable_scope(name) as scope:
        kernel = _variable_with_weight_decay('weights',
                                             shape=kernel_shape, wd=wd)
        conv = tf.nn.conv2d(x, kernel, strides, padding='SAME')
    return conv


def batch_norm(name, is_training, x):
    return tf.contrib.layers.batch_norm(inputs=x,
                                        center=True, scale=True,
                                        is_training=is_training,
                                        scope=name)


def gloabal_avg(x):
    assert x.get_shape().ndims == 4
    return tf.reduce_mean(x, [1, 2])


def fully_layer(x, out_node):
    dim = x.get_shape()[-1].value
    W = _variable_with_weight_decay('weights', [dim, out_node])
    b = tf.get_variable('biases', [out_node],
                        initializer=tf.constant_initializer(0))
    return tf.nn.xw_plus_b(x, W, b)


def residual_block(x, in_channels, out_channels, strides, is_training=True, activate_before_residual=False):
    if activate_before_residual:
        with tf.variable_scope('activate_before_residual'):
            x = batch_norm('bn0', is_training, x)
            x = tf.nn.relu(x)
            shortcut_x = x
    else:
        with tf.variable_scope('residual_only_activation'):
            shortcut_x = x
            x = batch_norm('bn1', is_training, x)
            x = tf.nn.relu(x)
    # sub1
    with tf.variable_scope('conv1'):
        x = conv_layer('conv1', x, [3, 3, in_channels, out_channels], strides)
    # sub2
    with tf.variable_scope('conv2'):
        x = batch_norm('bn2', is_training, x)
        x = tf.nn.relu(x)
        x = conv_layer(
            'conv2', x, [3, 3, out_channels, out_channels], strides=[1, 1, 1, 1])

    with tf.variable_scope('add_shortcut_x'):
        if in_channels != out_channels:
            shortcut_x = tf.nn.avg_pool(
                shortcut_x, strides, strides, padding='SAME')
            shortcut_x = tf.pad(shortcut_x, [[0, 0], [0, 0], [0, 0],
                                             [(out_channels - in_channels) // 2, (out_channels - in_channels) // 2]])
        x += shortcut_x
    return x

def _activation_summary(x,multi_gpu=True):
    if multi_gpu:
        name=x.op.name[8:]
        tf.summary.histogram(name, x)


def inference(x, num_units_per_block, is_training):
    with tf.variable_scope('init'):
        x = conv_layer('conv_0', x, [3, 3, 3, 16], [1, 1, 1, 1])

    with tf.variable_scope('residual_block_1_0'):
        x = residual_block(x, 16, 16, [1, 1, 1, 1], is_training, True)

    for i in range(1, num_units_per_block):
        with tf.variable_scope('residual_block_1_%d' % i):
            x = residual_block(x, 16, 16, [1, 1, 1, 1], is_training, False)

    with tf.variable_scope('residual_block_2_0'):
        x = residual_block(x, 16, 32, [1, 2, 2, 1], is_training, False)

    for i in range(1, num_units_per_block):
        with tf.variable_scope('residual_block_2_%d' % i):
            x = residual_block(x, 32, 32, [1, 1, 1, 1], is_training, False)

    with tf.variable_scope('residual_block_3_0'):
        x = residual_block(x, 32, 64, [1, 2, 2, 1], is_training, False)

    for i in range(1, num_units_per_block):
        with tf.variable_scope('residual_block_3_%d' % i):
            x = residual_block(x, 64, 64, [1, 1, 1, 1], is_training, False)

    with tf.variable_scope('gloabal_avg'):
        x = batch_norm('bn_0', is_training, x)
        x = tf.nn.relu(x)
        x = gloabal_avg(x)
        dim=x.get_shape()[-1].value
        x = tf.reshape(x, [-1,dim])

    with tf.variable_scope('softmax'):
        logits = fully_layer(x, 10)
        _activation_summary(logits)

    return logits


def loss(logits, labels):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                            labels=labels, name='cross_entropy_loss')
    cross_entropy_mean = tf.reduce_mean(
        cross_entropy, name='cross_entropy_mean_loss')
    tf.add_to_collection('losses', cross_entropy_mean)
    total_loss = tf.add_n(tf.get_collection('losses'), name='total_loss')
    
    return total_loss


def accurracy(logits, labels):
    predition = tf.nn.softmax(logits)
    acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(predition, axis=1),
                                          tf.argmax(labels, axis=1)), 'float'))
    return acc


def error(logits, labels):
    err = 1 - accurracy(logits, labels)
    return err

