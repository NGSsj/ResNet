# -*- coding: utf-8 -*-
# Date    : 2017-09-07 12:47:54
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

import tensorflow as tf 
import numpy as np 
import resnet
import math
import input_data

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', './logs/train/',
                           """Directory where to write checkpoint and training event of acc ,error.""")
tf.app.flags.DEFINE_integer('num_test', 10000, 'num of test images')

tf.app.flags.DEFINE_integer('BATCH_SIZE', 100,
                            """Number of images to process in a batch.""")

tf.app.flags.DEFINE_string('DATA_DIR', './cifar10_data/cifar-10-batches-bin',
                           """Path to the CIFAR-10 data directory.""")

def num_correct_predition(logits,labels):
    predition=tf.nn.softmax(logits)
    correct = tf.equal(tf.arg_max(predition, 1), tf.arg_max(labels, 1))
    correct=tf.cast(correct, tf.int32)
    return tf.reduce_sum(correct)

def evaluate():
	with tf.Graph().as_default():
		test_images, test_labels = input_data.read_cifar10(FLAGS.DATA_DIR, 
			False, FLAGS.BATCH_SIZE, False)
		logits=resnet.inference(test_images, 7, is_training=False)
		init=tf.global_variables_initializer()
		acc_num=num_correct_predition(logits, test_labels)
		saver=tf.train.Saver()
		with tf.Session() as sess:
			sess.run(init)
			coord = tf.train.Coordinator()
			threads = tf.train.start_queue_runners(sess = sess, coord = coord)
			print('reading checkpoints...')
			checkpoint=tf.train.latest_checkpoint(FLAGS.train_dir)
			if checkpoint:
				saver.restore(sess, checkpoint)
				print("restore from the checkpoint {0}".format(checkpoint))
			else:
				print('No checkpoint file found')
			try:
				print('\nEvaluating...')
				num_step=int(math.floor(FLAGS.num_test/FLAGS.BATCH_SIZE))
				num_sample=num_step*FLAGS.BATCH_SIZE
				step=0
				total_correct=0
				print(num_step,num_sample)
				while step<num_step:
					batch_correct=sess.run(acc_num)
					total_correct+=np.sum(batch_correct)
					step+=1
					print(step)
				print('Total testing samples: %d' %num_sample)
				print('Total correct predictions: %d' %total_correct)
				print('Average accuracy: %.2f%%' %(100*total_correct/num_sample))
			except Exception as e:
				pass
			finally:
				coord.request_stop()
				coord.join(threads)

if __name__ == '__main__':
	evaluate()