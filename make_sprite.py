import os
import sys
import time
import json
import tensorflow as tf

import pdb

from tensorflow.python.ops import control_flow_ops

from datetime import datetime

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from model.vaegan import DCGAN
from iohandler.datareader import img_reader


# datadir = '/home/jrm/proj/Hanzi/TWKai98_32x32'
datadir = 'sprite_imgs'

n = 100

coord = tf.train.Coordinator()
arch = json.load(open('architecture.json'))
imgs, info = img_reader(
    datadir=datadir,
    img_dims=(arch['img_h'], arch['img_w'], arch['img_c']),
    batch_size=n*n,
    rtype='tanh',
    num_threads=1,
    shuffle=False)


# imgs_NxD = tf.reshape(imgs, [-1, 32*32])
imgs_10kx32x32x1 = imgs

imgs = tf.reshape(imgs, [n, n, 32, 32])
imgs = tf.transpose(imgs, [0, 2, 1, 3])
imgs = tf.reshape(imgs, [32*n, 32*n])


sess = tf.Session()
# init = tf.initialize_all_variables()
init = tf.global_variables_initializer()
sess.run(init)

threads = tf.train.start_queue_runners(sess=sess, coord=coord)


imgs_np = sess.run(imgs_10kx32x32x1)
import pdb
pdb.set_trace()
imgs_np.tofile('sprite-10000x32x32x1')


# plt.figure()
# plt.imshow(images, cmap='gray')
# plt.savefig('sprite.jpeg')

# from PIL import Image
# i = (images / 2 + 0.5) * 255
# i = i.astype(np.uint8)

# im = Image.fromarray(i)
# im.save('sprite.png')

import shutil
import os
import numpy as np

from PIL import Image

images = sess.run(imgs)
i = (images / 2 + 0.5) * 255
i = i.astype(np.uint8)

im = Image.fromarray(i)
im.save('sprite.jpg')

# import numpy as np
# from PIL import Image
# from iohandler.datareader import find_files
# datadir = 'sprite_imgs'
# files = find_files(datadir, pattern='.*\.jpg')
# files = sorted(files)
# imgs = []
# for f in files:
# 	i = Image.open(f)
# 	i = np.reshape(i, [1, 32, 32])
# 	imgs.append(i)







# # 
# img_dims = (32, 32, 1)

# with tf.variable_scope('input'):
# 	# [TODO] should I merge tf.train.string_input_producer with `find_files`?
# 	h, w, c = img_dims
# 	filename_queue = tf.train.string_input_producer(files)
# 	reader = tf.WholeFileReader()
# 	key, value = reader.read(filename_queue)
# 	img = decoder(value, channels=c)
# 	img = tf.image.crop_to_bounding_box(img, 0, 0, h, w)
# 	img = tf.to_float(img)
# 	if rtype == 'tanh':
# 		img = tf.div(img, 127.5) - 1.
# 	elif rtype == 'sigmoid':
# 		img = tf.div(img, 255.)
# 	else:
# 		raise ValueError(
# 			'Unsupported range type: {:s}.'.format(rtype) + 
# 			'(sigmoid or tanh)')
# 	img = tf.expand_dims(img, 0)


import shutil
from iohandler.datareader import find_files
datadir = '/home/jrm/proj/Hanzi/TWKai98_32x32'
files = find_files(datadir, pattern='.*\.jpg')
for i in range(10000):
	shutil.copy(files[i], 'sprite_imgs/')

