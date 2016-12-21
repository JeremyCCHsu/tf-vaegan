# -*- coding: utf-8 -*-
import os
import re
import sys
import time
import json

import pdb
import numpy as np
import tensorflow as tf

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from scipy.io import loadmat
from datetime import datetime

from model.vaegan import VAEGAN

from PIL import Image
from iohandler.datareader import find_files

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string(
    'datadir', './data/TWKai98_32x32', 'data dir')
tf.app.flags.DEFINE_string(
    'architecture', None, 'network architecture')
tf.app.flags.DEFINE_string('logdir', 'logdir', 'log dir')
tf.app.flags.DEFINE_string('checkpoint', None, 'model checkpoint')


def SingleFileReader(filename, shape, rtype='tanh', ext='jpg'):    
    n, h, w, c = shape
    if ext == 'jpg' or ext == 'jpeg':
        decoder = tf.image.decode_jpeg
    elif ext == 'png':
        decoder = tf.image.decode_png
    else:
        raise ValueError('Unsupported file type: {:s}.'.format(ext) + 
            ' (only *.png and *.jpg are supported')

    filename_queue = tf.train.string_input_producer(filename, shuffle=False)
    reader = tf.WholeFileReader()
    key, value = reader.read(filename_queue)
    img = decoder(value, channels=c)
    img = tf.image.crop_to_bounding_box(img, 0, 0, h, w)
    img = tf.to_float(img)
    if rtype == 'tanh':
        img = tf.div(img, 127.5) - 1.

    imgs = tf.train.batch(
        [img],
        batch_size=n,
        capacity=1)
    return imgs, key


def fit_the_shape(x_, shape):
    n, h, w, c = shape
    x_ = np.reshape(
        np.transpose(x_, [1, 0, 2, 3]),
        [h, w * n, c])
    if x_.shape[-1] == 1:
        x_ = x_[:, :, 0]     
    return x_


def main():
    if FLAGS.checkpoint is None:
        raise ValueError('You must specify a checkpoint file.')

    # FLAGS
    started_datestring = "{0:%Y-%m%d-%H%M-%S}".format(datetime.now())
    logdir = os.path.join(FLAGS.logdir, 'generate', started_datestring)

    if FLAGS.architecture is None:
        ckpt_dir = os.path.split(FLAGS.checkpoint)[0]
        architecture = os.path.join(ckpt_dir, 'architecture.json')
    else:
        architecture = FLAGS.architecture
    
    with open(architecture) as f:
        arch = json.load(f)

    h, w, c = arch['hwc']

    coord = tf.train.Coordinator()

    print(FLAGS.datadir)

    net = VAEGAN(arch, is_training=False)

    n = 3
    filenames = list()
    with open('test.txt', encoding='utf8') as f:
        for line in f:
            chars = list(line.strip())
            for char in chars:
                filename = os.path.join(
                    FLAGS.datadir,
                    'U{:d}.jpg'.format(ord(char)))
                filenames.append(filename)
    n_iter = len(filenames) // n

    xs, _ = SingleFileReader(filenames, shape=[n, h, w, c])
    z = net.encode(xs)['mu']
    xh = net.decode(z, tanh=True)

    z_any = tf.placeholder(dtype=tf.float32, shape=[None, arch['z_dim']])
    xh_any = net.decode(z_any, tanh=True)

    # Restore model
    sess = tf.Session()
    saver = tf.train.Saver()
    print('Restoring model from {}'.format(FLAGS.checkpoint))
    saver.restore(sess, FLAGS.checkpoint)

    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    try:
        for it in range(n_iter):

            x_, z_, xh_ = sess.run([xs, z, xh])

            z_final = z_[0] - z_[1] + z_[2]
            z_final = np.reshape(z_final, [1, -1])

            a = np.asarray([.25, .5, 1., 2., 3]).reshape([-1, 1])
            x_f_all = sess.run(xh_any, feed_dict={z_any: a * z_final})

            x_refer = x_[0] - x_[1] + x_[2]

            xr = x_refer[:, :, 0]

            plt.figure()
            plt.subplot(4, 1, 1)
            plt.imshow(
                fit_the_shape(x_, [n, h, w, c]),
                cmap='gray')
            plt.subplot(4, 1, 2)
            plt.imshow(
                fit_the_shape(xh_, [n, h, w, c]),
                cmap='gray')
            plt.subplot(4, 1, 3)
            plt.imshow(
                fit_the_shape(x_f_all, [len(a), h, w, c]),
                cmap='gray')
            plt.subplot(4, 1, 4)
            plt.imshow(xr, cmap='gray')
            plt.savefig('test-arith-{}.png'.format(
                # ''.join(chars[3 * it: 3 * (it + 1)])))
                it))
            plt.close()

    finally:
        coord.request_stop()
        coord.join(threads)


if __name__ == '__main__':
    main()
