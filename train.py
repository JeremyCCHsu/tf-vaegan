import os
import sys
import time
import json
import tensorflow as tf

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from model.vaegan import DCGAN
from iohandler.datareader import img_reader


logdir = 'log'
lr = 2e-4
beta1 = 0.5
datadir = '/home/jrm/proj/Hanzi/TWKai98_32x32'
n_epoch = 100
n_step = 16300
batch_size = 64




def save(saver, sess, logdir, step):
    model_name = 'model.ckpt'
    checkpoint_path = os.path.join(logdir, model_name)
    print('Storing checkpoint to {} ...'.format(logdir), end="")
    sys.stdout.flush()

    if not os.path.exists(logdir):
        os.makedirs(logdir)

    saver.save(sess, checkpoint_path, global_step=step)
    print(' Done.')


# x = tf.placeholder(tf.float32, shape=[128, 32, 32, 1])

coord = tf.train.Coordinator()
imgs, info = img_reader(
	datadir=datadir,
	img_dims=(32, 32, 1),
	batch_size=batch_size,
	rtype='tanh')
	# num_examples_per_epoch=20941)

step_per_ep = info['n_files'] // batch_size
# g = sess.run(imgs)

arch = json.load(open('architecture.json'))

machine = DCGAN(arch, is_training=True)

loss = machine.loss(imgs)
xh = machine.sample(batch_size)

trainables = tf.trainable_variables()
g_vars = [v for v in trainables if 'Generator' in v.name]
d_vars = [v for v in trainables if 'Discriminator' in v.name]

# # print(g_vars)
# print('\nGenerator')
# for v in g_vars:
# 	print(v.name)
# print('\n\n\nDiscriminator')
# # print(d_vars)
# for v in d_vars:
# 	print(v.name)

# optimizer = tf.train.AdamOptimizer(lr)
obj_D = loss['D_fake'] + loss['D_real']

# optimizer = tf.train.AdamOptimizer(lr, beta1)
optimizer = tf.train.RMSPropOptimizer(lr)

opt_d = optimizer.minimize(
	obj_D,
	var_list=d_vars)
opt_g = optimizer.minimize(
	loss['G_fake'],
	var_list=g_vars)


writer = tf.train.SummaryWriter(logdir)
writer.add_graph(tf.get_default_graph())

sess = tf.Session()
init = tf.initialize_all_variables()
sess.run(init)

saver = tf.train.Saver()

threads = tf.train.start_queue_runners(sess=sess, coord=coord)


# for epoch in range(n_epoch):
try:

	n_iter = info['n_files'] // batch_size
	time_i = time.time()
	for ep in range(n_epoch):
		for it in range(n_iter):

			_, l_df, l_dr = sess.run([opt_d, loss['D_fake'], loss['D_real']])
			_, l_g = sess.run([opt_g, loss['G_fake']])
			t = time.time() - time_i
			print('Epoch [{:3d}/{:3d}] [{:4d}/{:4d}] d_loss={:6.3f}+{:6.3f}, g_loss={:5.2f}, T={:.2f}'
				.format(ep, n_epoch, it, n_iter, l_df, l_dr, l_g, t))	

			if it % (n_iter // 2) == 0:	
				creation = sess.run(xh)
				real = sess.run(imgs)
	
				# [TODO] functionize
				plt.figure()
				for i in range(15):
					plt.subplot(4, 4, i + 1)
					plt.imshow(creation[i, :, :, 0], interpolation='none', cmap='gray')
					plt.colorbar()
				plt.subplot(4, 4, 16)
				plt.imshow(real[0, :, :, 0], interpolation='none', cmap='gray')
				plt.colorbar()
				plt.savefig('test-Ep{:03d}-It{:04d}.png'.format(ep, it))
				plt.close()


except KeyboardInterrupt:
	print()

finally:
	save(saver, sess, logdir, ep*n_iter + it)
	coord.request_stop()
	coord.join(threads)

