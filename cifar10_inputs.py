# -*- coding: utf-8 -*-
# Date    : 2017-09-05 17:58:42
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow.python.platform
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

from tensorflow.python.platform import gfile

# Global constants describing the CIFAR-10 data set.
IMAGE_SIZE = 32
NUM_CLASSES = 10
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000


def read_cifar10(filename_queue):
  class CIFAR10Record(object):
    pass
  result = CIFAR10Record()
  label_bytes = 1  # 2 for CIFAR-100
  result.height = 32
  result.width = 32
  result.depth = 3
  image_bytes = result.height * result.width * result.depth
  record_bytes = label_bytes + image_bytes

  reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
  result.key, value = reader.read(filename_queue)
  record_bytes = tf.decode_raw(value, tf.uint8)
  result.label = tf.cast(
      tf.slice(record_bytes, [0], [label_bytes]), tf.int32)
  depth_major = tf.reshape(tf.slice(record_bytes, [label_bytes], [image_bytes]),
                           [result.depth, result.height, result.width])
  result.uint8image = tf.transpose(depth_major, [1, 2, 0])

  return result


def _generate_image_and_label_batch(image, label, min_queue_examples, batch_size):
  num_preprocess_threads = 16
  images, label_batch = tf.train.shuffle_batch([image, label], batch_size=batch_size,
                                               num_threads=num_preprocess_threads,
                                               capacity=min_queue_examples + 3 * batch_size,
                                               min_after_dequeue=min_queue_examples)

  labels=tf.one_hot(label_batch, 10)
  labels=tf.cast(labels, tf.int32)
  labels=tf.reshape(labels, [batch_size,10])
  return images, labels
  

def inputs(eval_data, data_dir, batch_size):
  if not eval_data:
    filenames = [os.path.join(data_dir, 'data_batch_%d.bin' % i)
                 for i in xrange(1, 6)]
    num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
    for f in filenames:
      if not gfile.Exists(f):
        raise ValueError('Failed to find file: ' + f)
    filename_queue = tf.train.string_input_producer(filenames)
    read_input = read_cifar10(filename_queue)
    reshaped_image = tf.cast(read_input.uint8image, tf.float32)
    resized_image = tf.image.resize_image_with_crop_or_pad(reshaped_image,
                                                           IMAGE_SIZE + 4, IMAGE_SIZE + 4)
    crop_image = tf.random_crop(resized_image, [IMAGE_SIZE, IMAGE_SIZE, 3])
    image = tf.image.random_flip_left_right(crop_image)
    image = tf.image.per_image_standardization(image)

    label = read_input.label
   
    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(num_examples_per_epoch *
                             min_fraction_of_examples_in_queue)

    return _generate_image_and_label_batch(image, label,
                                           min_queue_examples, batch_size)
  else:
    filenames = [os.path.join(data_dir, 'test_batch.bin')]
    num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_EVAL
    for f in filenames:
      if not gfile.Exists(f):
        raise ValueError('Failed to find file: ' + f)
      filename_queue = tf.train.string_input_producer(filenames)
      read_input = read_cifar10(filename_queue)
      reshaped_image = tf.cast(read_input.uint8image, tf.float32)
      image = tf.image.per_image_standardization(reshaped_image)

      label = read_input.label
      
      min_fraction_of_examples_in_queue = 0.4
      min_queue_examples = int(num_examples_per_epoch *
                             min_fraction_of_examples_in_queue)

      
      return _generate_image_and_label_batch(image, label,
                                           min_queue_examples, batch_size)
