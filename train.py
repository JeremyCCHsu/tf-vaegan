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
datadir = '/home/jrm/proj/Hanzi/TWKai98_32x32'
n_epoch = 10
n_step = 16300
batch_size = 64
beta1 = 0.5



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
	batch_size=64,
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

opt_d = tf.train.AdamOptimizer(lr, beta1).minimize(
	obj_D,
	var_list=d_vars)
opt_g = tf.train.AdamOptimizer(lr, beta1).minimize(
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
	for step in range(n_step):
		time_i = time.time()
		_, l_df, l_dr = sess.run([opt_d, loss['D_fake'], loss['D_real']])
		_, l_g = sess.run([opt_g, loss['G_fake']])
		print('Step {:4d}, d_loss: {:.4f}+{:.4f}, g_loss: {:.4f}, time={:.2f}'
			.format(step, l_df, l_dr, l_g, time.time() - time_i))

		if step % step_per_ep == 0:	
			creation = sess.run(xh)
			real = sess.run(imgs)

			plt.figure()
			for i in range(15):
				plt.subplot(4, 4, i + 1)
				plt.imshow(creation[i, :, :, 0], interpolation='none', cmap='gray')
				plt.colorbar()
			plt.subplot(4, 4, 16)
			plt.imshow(real[0, :, :, 0], interpolation='none', cmap='gray')
			plt.colorbar()
			plt.savefig('test-{:2d}.png'.format(step//step_per_ep))
			plt.close()


except KeyboardInterrupt:
	print()

finally:
	save(saver, sess, logdir, step)
	coord.request_stop()
	coord.join(threads)

