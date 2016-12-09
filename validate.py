# -*- coding: utf-8 -*-

import os
import re
import sys
import time
import json

import pdb
import numpy as np
import tensorflow as tf
# [TODO]
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from scipy.io import loadmat
from datetime import datetime

from model.vaegan import DCGAN

from PIL import Image

# from model.vae import VAE

# from iohandler.spectral_reader import vc2016TFWholeReader, DSP
# from util.spectral_processing import MelCepstralProcessing

# from util.layers import GaussianKLD

FLAGS = tf.app.flags.FLAGS
# tf.app.flags.DEFINE_string('source', 'SF1', 'list of source speakers')
# tf.app.flags.DEFINE_string('target', 'TM3', 'list of target speakers')
# tf.app.flags.DEFINE_string('{:s}-{:s}-trn', '', 'data dir')
tf.app.flags.DEFINE_string(
    'datadir', '/home/jrm/proj/Hanzi/TWKai98_32x32', 'data dir')
tf.app.flags.DEFINE_string(
    'architecture', 'architecture.json', 'network architecture')
tf.app.flags.DEFINE_string('logdir', 'logdir', 'log dir')
# tf.app.flags.DEFINE_string(
#     'logdir_root', None, 'log dir')
# tf.app.flags.DEFINE_string(
#     'restore_from', None, 'resotre form dir')
tf.app.flags.DEFINE_string('checkpoint', None, 'model checkpoint')

tf.app.flags.DEFINE_integer('batch_size', 128, 'batch size')
tf.app.flags.DEFINE_float('l2_regularization', 0.0, 'L2 regularization')
tf.app.flags.DEFINE_float('lr', 1e-3, 'learning rate')
tf.app.flags.DEFINE_integer('num_steps', 10000, 'num of steps (frames)')

tf.app.flags.DEFINE_integer('source_id', 0, 'target id (SF1 = 1, TM3 = 9)')
tf.app.flags.DEFINE_integer('target_id', 9, 'target id (SF1 = 1, TM3 = 9)')

tf.app.flags.DEFINE_string(
    'file_filter', '.*\.bin', 'filename filter')

# TEST_PATTERN = 'SF1-100001.bin'
TEST_PATTERN = '.*001.bin'
N_SPEAKER = 10

mFea = 513 + 1

# ['SF1', 'SF2', 'SF3', 'SM1', 'SM2', 'TF1', 'TF2', 'TM1', 'TM2', 'TM3']

# from iohandler import DSP



def SingleFileReader(filename, shape, rtype='tanh', ext='jpg'):    
    h, w, c = shape
    if ext == 'jpg' or ext == 'jpeg':
        decoder = tf.image.decode_jpeg
    elif ext == 'png':
        decoder = tf.image.decode_png
    else:
        raise ValueError('Unsupported file type: {:s}.'.format(ext) + 
            ' (only *.png and *.jpg are supported')

    filename = tf.train.string_input_producer(filename)
    reader = tf.WholeFileReader()
    key, value = reader.read(filename)
    img = decoder(value, channels=c)
    img = tf.image.crop_to_bounding_box(img, 0, 0, h, w)
    img = tf.to_float(img)
    if rtype == 'tanh':
        img = tf.div(img, 127.5) - 1.

    imgs = tf.train.batch(
        [img],
        batch_size=1,
        capacity=1)
    return imgs, key



def main():
    if FLAGS.checkpoint is None:
        raise

    # FLAGS
    started_datestring = "{0:%Y-%m%d-%H%M-%S}".format(datetime.now())
    logdir = os.path.join(FLAGS.logdir, 'generate', started_datestring)
    # with open(FLAGS.)

    # architecture = os.path.join(
    #     os.path.split(FLAGS.checkpoint)[0],
    #     'architecture.json')
    # print(architecture)
    # # pdb.set_trace()

    with open(FLAGS.architecture) as f:
        arch = json.load(f)

    coord = tf.train.Coordinator()

    print(FLAGS.datadir)
    # label, spectrum, filename = vc2016TFWholeReader(
    #     datadir=FLAGS.datadir,
    #     pattern=TEST_PATTERN,
    #     output_filename=True)


    net = DCGAN(arch, is_training=False)

    with open('test.txt', encoding='utf8') as f:
        char = [c for line in f for c in line]

    print(char)

    num = map(ord, char)
    
    # [print(n) for n in num]

    # print(num)

    # # n = [n for n in num]
    # xs = [SingleFileReader(
    #         [os.path.join(FLAGS.datadir, 'U{:d}.jpg'.format(n))],
    #         shape=[32, 32, 1]) for n in num]
    xs = list()
    for n in num:
        x, _ = SingleFileReader(
            [os.path.join(FLAGS.datadir, 'U{:d}.jpg'.format(n))],
            shape=[32, 32, 1])
        xs.append(x)


    # x1, _ = SingleFileReader(
    #     [os.path.join(FLAGS.datadir, 'U28165.jpg')],
    #     shape=[32, 32, 1])

    # x2, _ = SingleFileReader(
    #     [os.path.join(FLAGS.datadir, 'U35531.jpg')],
    #     shape=[32, 32, 1])

    # x3, _ = SingleFileReader(
    #     [os.path.join(FLAGS.datadir, 'U35441.jpg')],
    #     shape=[32, 32, 1])


    x = tf.placeholder(dtype=tf.float32, shape=[None, 32, 32, 1])

    z = net.encode(x)['mu']

    xh = net.decode(z)

    # processor = DSP(
    #     xmin=np.load('test_tmp/xmin.npf.npy').astype(np.float32),
    #     xmax=np.load('test_tmp/xmax.npf.npy').astype(np.float32))

    # x_logspec = tf.placeholder(tf.float32, shape=[None, architecture['n_x']])
    # y_ = tf.placeholder(tf.float32, shape=[None, architecture['n_y']])

    # x_ = processor.forward_process(x_logspec)
    # z_ = net.encode(x_)
    # z_mu_, z_lv_ = net.encode_zuv(x_)
    # xh_ = net.decode(z_, y_)
    # xh_ = processor.backward_process(xh_)


    # kld = tf.reduce_mean(
    #     GaussianKLD(
    #         z_mu_,
    #         z_lv_,
    #         tf.zeros_like(z_mu_),
    #         tf.zeros_like(z_lv_),
    #         dimwise=True),
    #     0)

    # Restore model
    sess = tf.Session()


    # variables_to_restore = {
    #     var.name[:-2]: var for var in tf.all_variables()
    #     if not ('state_buffer' in var.name or 'pointer' in var.name)}
    # saver = tf.train.Saver(variables_to_restore)
    saver = tf.train.Saver()

    print('Restoring model from {}'.format(FLAGS.checkpoint))
    saver.restore(sess, FLAGS.checkpoint)
    # threads = tf.train.start_queue_runners(sess=sess, coord=coord)


    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    try:
        
        x1 = sess.run(xs[0])
        x2 = sess.run(xs[1])
        x3 = sess.run(xs[2])
        # [sess.run(v) for v in ]

        z1 = sess.run(z, feed_dict={x: x1})
        z2 = sess.run(z, feed_dict={x: x2})
        z3 = sess.run(z, feed_dict={x: x3})

        xh1 = sess.run(xh, feed_dict={z: z1})
        xh2 = sess.run(xh, feed_dict={z: z2})
        xh3 = sess.run(xh, feed_dict={z: z3})


        z_final = z1 - z2 + z3

        x_f_all = list()
        for a in [.25, .5, 1., 2., 3.]:
            x_final = sess.run(xh, feed_dict={z: a * z_final})
            x_f_all.append(x_final[0, :, :, 0])
        # x_f_all = np.transpose(np.asarray(x_f_all), [1, 2, 0]

        x_refer = x1 - x2 + x3

        xr = x_refer[0, :, :, 0]

        plt.figure()
        plt.subplot(4, 1, 1)
        plt.imshow(
            np.concatenate([x1[0,:,:,0], x2[0,:,:,0], x3[0,:,:,0]],
                axis=1),
            cmap='gray')
        plt.subplot(4, 1, 2)
        plt.imshow(
            np.concatenate([xh1[0,:,:,0], xh2[0,:,:,0], xh3[0,:,:,0]],
                axis=1),
            cmap='gray')
        plt.subplot(4, 1, 3)
        # plt.imshow(x_final[0, :, :, 0], cmap='gray')
        plt.imshow(
            np.concatenate(x_f_all, axis=1), cmap='gray')
        # plt.subplot(4, 1, 4)
        # plt.imshow(
        #     np.concatenate(
        #         [abs(xr), xr],
        #         axis=1),
        #     cmap='gray')
        plt.subplot(4, 1, 4)
        plt.imshow(xr, cmap='gray')
        plt.savefig('test-arith{}.png'.format(''.join(char)))
        plt.close()
        # pdb.set_trace()

    finally:
        # save(saver, sess, dirs['logdir'], step)
        coord.request_stop()
        coord.join(threads)



if __name__ == '__main__':
    main()
