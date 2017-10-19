# -*- coding: utf-8 -*-
# Date    : 2017-09-08 10:23:29
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
import resnet
import numpy as np
import os
import input_data
import re

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', './multi-gpu-logs/train/',
                           """Directory where to write checkpoint and training acc ,error.""")
tf.app.flags.DEFINE_integer('max_steps', 80000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('is_training', True,
                            """Whether to use Batch Normalization or not""")
tf.app.flags.DEFINE_integer('num_units_per_block',
                            7, """num_units_per_block""")
tf.app.flags.DEFINE_integer('num_gpus', 2,
                            """How many GPUs to use.""")
tf.app.flags.DEFINE_float('lr', 0.1, """lerning rate""")
tf.app.flags.DEFINE_integer('batch_size', 64, "batch size images for training...")
tf.app.flags.DEFINE_string('data_dir', './cifar10_data/cifar-10-batches-bin',
                           """Path to the CIFAR-10 data directory.""")
tf.app.flags.DEFINE_string('TOWER_NAME', 'tower', """tower name...""")


def tower_loss(scope):
    images, labels = input_data.read_cifar10(FLAGS.data_dir,True, FLAGS.batch_size, True)
    logits=resnet.inference(images, FLAGS.num_units_per_block, FLAGS.is_training)
    _=resnet.loss(logits, labels)
    losses=tf.get_collection('losses',scope)
    total_loss=tf.add_n(losses,name='total_loss')
    with tf.name_scope(None) as scope:
        tf.summary.scalar("total_loss", total_loss)
    return total_loss

def average_gradient(tower_grads):
    average_grads=[]
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
    #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads=[]
        for g,_ in grad_and_vars:
            expanded_g=tf.expand_dims(g,0)
            grads.append(expanded_g)
        grad_tensors=tf.concat(grads, axis=0)
        grad=tf.reduce_mean(grad_tensors,axis=0)
        var=grad_and_vars[0][1]
        grad_and_var=(grad,var)
        average_grads.append(grad_and_var)
    return average_grads


def train():
    with tf.Graph().as_default(),tf.device('/cpu:0'):
        global_step=tf.Variable(0,trainable=False)
        learning_rate=tf.train.exponential_decay(FLAGS.lr, global_step, 32000, 0.1,staircase=False)
        tf.summary.scalar('learninig_rate', learning_rate)
        opt=tf.train.GradientDescentOptimizer(learning_rate)
        tower_grads=[]
        for i in range(FLAGS.num_gpus):
            with tf.device('/gpu:%d'%i):
                with tf.name_scope('%s_%d'%(FLAGS.TOWER_NAME,i)) as scope:
                    loss=tower_loss(scope)
                    tf.get_variable_scope().reuse_variables()
                    # summaries=tf.get_collection(tf.GraphKeys.SUMMARIES)
                    grads=opt.compute_gradients(loss)
                    tower_grads.append(grads)
        grads_and_vars=average_gradient(tower_grads)
        update_ops=tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            apply_gradient_op=opt.apply_gradients(grads_and_vars,global_step=global_step)
        summary_op=tf.summary.merge_all()
        saver=tf.train.Saver(tf.all_variables())
        init_op=tf.global_variables_initializer()
        config=tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            coord=tf.train.Coordinator()
            threads=tf.train.start_queue_runners(sess=sess,coord=coord)
            train_summary_writer=tf.summary.FileWriter(FLAGS.train_dir,sess.graph)
            sess.run(init_op)
            start_step=0
            checkpoint=tf.train.latest_checkpoint(FLAGS.train_dir)
            if checkpoint:
                saver.restore(sess, checkpoint)
                print("restore from the checkpoint {0}".format(checkpoint))
                start_step+=int(checkpoint.split('-')[-1])
            print("start training...")
            try:
                for step in range(start_step,FLAGS.max_steps):
                    sess.run(apply_gradient_op)
                    if step%100==0 or step==(FLAGS.max_steps-1) :
                        tra_los=sess.run(loss)
                        print('Step: %d, loss: %.6f'%(step,tra_los))
                    if step%200==0 or step==(FLAGS.max_steps-1):
                        summary_str=sess.run(summary_op)
                        train_summary_writer.add_summary(summary_str,step)
                    if step%2000==0 or step==(FLAGS.max_steps-1):
                        checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
                        saver.save(sess, checkpoint_path,global_step=step)
            except tf.errors.OutOfRangeError:
                coord.request_stop()
            finally:
                coord.request_stop()
                coord.join(threads)

if __name__ == '__main__':
    train()
    

    




            
            