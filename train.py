import tensorflow as tf
import resnet
import numpy as np
import os
import input_data

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', './logs/train/',
                           """Directory where to write checkpoint and training acc ,error.""")
tf.app.flags.DEFINE_string('test_dir', './logs/test/',
                           """Directory where to write test acc and err.""")
tf.app.flags.DEFINE_integer('max_steps', 80000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('is_training', True,
                            """Whether to use Batch Normalization""")
tf.app.flags.DEFINE_integer('num_units_per_block',
                            7, """num_units_per_block""")
tf.app.flags.DEFINE_float('lr', 0.1, """lerning rate""")
tf.app.flags.DEFINE_integer(
    'batch_size', 128, "batch size images for training...")
tf.app.flags.DEFINE_string('data_dir', './cifar10_data/cifar-10-batches-bin',
                           """Path to the CIFAR-10 data directory.""")


def train():
    with tf.name_scope('inputs'):
        train_images, train_labels = input_data.read_cifar10(
            FLAGS.data_dir, True, FLAGS.batch_size, True)
        test_images, test_labels = input_data.read_cifar10(
            FLAGS.data_dir, False, FLAGS.batch_size, False)
    xs = tf.placeholder(dtype=tf.float32, shape=(FLAGS.batch_size, 32, 32, 3))
    ys = tf.placeholder(dtype=tf.int32, shape=(FLAGS.batch_size, 10))
    global_step = tf.Variable(0, trainable=False)
    lerning_rate = tf.train.exponential_decay(
        FLAGS.lr, global_step, 32000, 0.1, staircase=False)
    tf.summary.scalar('lerning_rate', lerning_rate)

    logits = resnet.inference(xs, FLAGS.num_units_per_block,
                              FLAGS.is_training)
    loss = resnet.loss(logits, ys)
    tf.summary.scalar('loss', loss)
    opt = tf.train.MomentumOptimizer(lerning_rate, 0.9)
    update_op = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_op):
        train = opt.minimize(loss, global_step=global_step)

    acc_op = resnet.accurracy(logits, ys)
    tf.summary.scalar('accuracy', acc_op)
    err_op = resnet.error(logits, ys)
    tf.summary.scalar('error', err_op)

    summary_op = tf.summary.merge_all()
    saver = tf.train.Saver(tf.all_variables())
    init = tf.global_variables_initializer()
    coord = tf.train.Coordinator()

    with tf.Session() as sess:
        sess.run(init)
        train_summary_writer = tf.summary.FileWriter(
            FLAGS.train_dir, sess.graph)
        test_summary_writer = tf.summary.FileWriter(
            FLAGS.test_dir, sess.graph)
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        start_step = 0
        checkpoint = tf.train.latest_checkpoint(FLAGS.train_dir)
        if checkpoint:
            saver.restore(sess, checkpoint)
            print("restore from the checkpoint {0}".format(checkpoint))
            start_step += int(checkpoint.split('-')[-1])
        print("start training...")
        try:
            for step in range(start_step, FLAGS.max_steps):
                if coord.should_stop():
                    break
                tra_images_batch, tra_labels_batch = sess.run(
                    [train_images, train_labels])
                tes_images_batch, tes_labels_batch = sess.run(
                    [test_images, test_labels])
                _ = sess.run(train, feed_dict={
                             xs: tra_images_batch, ys: tra_labels_batch})
                if step % 50 == 0 or (step + 1) == FLAGS.max_steps:
                    tra_los, tra_acc = sess.run([loss, acc_op], feed_dict={
                                                xs: tra_images_batch, ys: tra_labels_batch})
                    print('Step: %d, loss: %.6f, accuracy: %.4f' %
                          (step, tra_los, tra_acc))
                if step % 200 == 0 or(step + 1) == FLAGS.max_steps:
                    tes_los, tes_acc = sess.run([loss, acc_op], feed_dict={
                                                xs: tes_images_batch, ys: tes_labels_batch})
                    print('***test_loss***Step: %d, loss: %.6f, accuracy: %.4f' %
                          (step, tes_los, tes_acc))
                if step % 300 == 0 or(step + 1) == FLAGS.max_steps:
                    summary_str1 = sess.run(summary_op, feed_dict={
                                            xs: tra_images_batch, ys: tra_labels_batch})
                    summary_str2 = sess.run(summary_op, feed_dict={
                                            xs: tes_images_batch, ys: tes_labels_batch})
                    train_summary_writer.add_summary(summary_str1, step)
                    test_summary_writer.add_summary(summary_str2, step)
                if step % 2000 == 0 or (step + 1) == FLAGS.max_steps:
                    checkpoint_path = os.path.join(
                        FLAGS.train_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=step)

        except tf.errors.OutOfRangeError:
            print('Done training -- epoch limit reached')
            coord.request_stop()
            coord.join()
        finally:
            coord.request_stop()
            coord.join(threads)


if __name__ == '__main__':
    train()
