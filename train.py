import os
import sys
import time
import json
import tensorflow as tf
import numpy as np

import pdb

# from tensorflow.python.ops import control_flow_ops

from datetime import datetime

# import matplotlib as mpl
# mpl.use('Agg')
# import matplotlib.pyplot as plt

from model.vaegan import VAEGAN
from iohandler.datareader import img_reader

from tensorflow.contrib.tensorboard.plugins import projector

from PIL import Image

from util.wrapper import save, load

STARTED_DATESTRING = datetime.now().strftime('%0m%0d-%0H%0M-%0S-%Y')
N_INTERP = 10
LOGDIR_ROOT = './logdir'
PATH_TO_SPRITE_IMAGE = './sprite/sprite.jpg'
PATH_TO_LABEL = './sprite/sprite-text.tsv'
SPRITE_NUMPY_FILE = './sprite/sprite-10000x32x32x1.npf'
N_VISUALIZE = 10000

args = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string(
    'datadir', './data/TWKai98_32x32', 'dir to dataset')
tf.app.flags.DEFINE_string(
    'logdir_root', None, 'root of log dir')
tf.app.flags.DEFINE_string(
    'logdir', None, 'log dir')
tf.app.flags.DEFINE_string(
    'restore_from', None, 'restore from dir (not from *.ckpt)')
tf.app.flags.DEFINE_string(
    'architecture', 'architecture.json', 'network architecture')
tf.app.flags.DEFINE_string(
    'gpu_cfg', None, 'GPU configuration')
tf.app.flags.DEFINE_integer('n_epoch', 100, 'num of epoch')
tf.app.flags.DEFINE_integer('batch_size', 64, 'batch size')
tf.app.flags.DEFINE_float('lr', 2e-4, 'learning rate')
tf.app.flags.DEFINE_float('beta1', 0.5, 'beta1 in AdamOptimizer')
tf.app.flags.DEFINE_float('reconst_v_gan', 1e-1, 'weight on DIS cost')
tf.app.flags.DEFINE_float('direct_sample', 0.0, 'weight on cost of direct samples')


def visualize_random_samples(sess, xh, n=8, filename=None):
    img_fake = sess.run(xh)

    shape = img_fake.shape[1:]
    shape = [n, n] + list(shape)
    img_fake = np.reshape(img_fake[: n*n], shape)
    img_fake = np.transpose(img_fake, [0, 2, 1, 3, 4])
    img_fake = np.reshape(
        img_fake,
        [shape[0]*shape[2], shape[0]*shape[3], shape[4]])
    img_fake = (img_fake - img_fake.min()) / (img_fake.max() - img_fake.min()) * 255
    img_fake = img_fake.astype(np.uint8)

    if filename:
        im = Image.fromarray(img_fake[:, :, 0])
        im.save(filename)


def visualize_interpolation(sess, x_interp, N=8, filename=None):
    x_s = list()
    for _ in range(1, N + 1):
        x = sess.run(x_interp)  # (n, h, w, c)
        x_s.append(x)
    x_s = np.concatenate(x_s)
    shapes = x_s.shape
    M = shapes[0] // N
    x_s = np.reshape(x_s, [N, M, shapes[1], shapes[2], shapes[3]])
    x_s = np.transpose(x_s, [0, 2, 1, 3, 4])
    x_s = np.reshape(x_s, [N * shapes[1], M * shapes[2], shapes[3]])
    x_s = (x_s / 2 + 0.5) * 255
    x_s = x_s.astype(np.uint8)
    # [TODO] 1. Use tf.image.encode_png, or
    #        2. deal with channel (Image.fromarray and plt.imshow support 2D only)
    im = Image.fromarray(x_s[:, :, 0])
    if filename:
        im.save(filename)


def get_optimization_ops(loss, args, mode='VAE-GAN'):
    '''
    [TODO]
    Although most of the trainer structures are the same,
    I think we have to use different training scripts for VAE- and DC-GAN
    (but do we have to have two different classes of VAE- and DC-?)
    '''
    optimizer = tf.train.AdamOptimizer(args.lr, args.beta1)

    trainables = tf.trainable_variables()
    g_vars = [v for v in trainables if 'Generator' in v.name]
    d_vars = [v for v in trainables if 'Discriminator' in v.name]

    if mode == 'DC-GAN':
        obj_D = loss['D_fake'] + loss['D_real']
        obj_G = loss['G_fake'] + loss['G_fake_xz'] 
        opt_e = None

    elif mode == 'VAE-GAN':
        e_vars = [v for v in trainables if 'Encoder' in v.name]

        obj_D = loss['D_fake'] + loss['D_real']
        obj_G = loss['G_fake'] \
            + loss['Dis'] * args.reconst_v_gan \
            # + loss['G_fake_xz'] * args.direct_sample
        obj_E = loss['KL(z)'] + loss['Dis']

        opt_e = optimizer.minimize(obj_E, var_list=e_vars)

    opt_d = optimizer.minimize(obj_D, var_list=d_vars)
    opt_g = optimizer.minimize(obj_G, var_list=g_vars)

    return opt_d, opt_g, opt_e


def get_default_logdir(logdir_root):
    return os.path.join(logdir_root, 'train', STARTED_DATESTRING)


def validate_log_dirs(args):
    if args.logdir and args.restore_from:
        raise ValueError(
            'You can only specify one of the following: ' +
            '--logdir and --restoreform')

    if args.logdir and args.log_root:
        raise ValueError('You can only specify either --logdir' +
            'or --logdir_root')

    if args.logdir_root is None:
        logdir_root = LOGDIR_ROOT

    if args.logdir is None:
        logdir = get_default_logdir(logdir_root)
        print('Using default logdir: {:s}'.format(logdir))

    # Note: `logdir` and `restore_from` are exclusive
    if args.restore_from is None:
        restore_from = logdir
    else:
        restore_from = args.restore_from 


    return dict(logdir=logdir,
        logdir_root=logdir_root,
        restore_from=restore_from)

# [TODO] load model for more work

def main():
    '''
    Note:
      1. The input is rescaled to [-1, 1] (img_reader: rtype)
    '''
    dirs = validate_log_dirs(args)

    coord = tf.train.Coordinator()

    with open(args.architecture) as f:
        arch = json.load(f)


    imgs, info = img_reader(
        datadir=args.datadir,
        img_dims=arch['hwc'],
        batch_size=args.batch_size,
        rtype='tanh')
    
    machine = VAEGAN(arch, is_training=True)

    loss = machine.loss(imgs)
    xh = machine.sample(args.batch_size)
    
    x_interp = machine.interpolate(imgs[0], imgs[1], N_INTERP)

    opt_d, opt_g, opt_e = get_optimization_ops(loss, args, arch['mode'])


    # # ========== For embedding =============
    # h, w, c = arch['hwc']
    # img4em = tf.Variable(
    #     np.reshape(
    #         np.fromfile(
    #             SPRITE_NUMPY_FILE, np.float32),
    #             [N_VISUALIZE, h, w, c]),
    #     name='emb_input_img')
    # codes = machine.encode(img4em)
    # em_var = tf.Variable(
    #     tf.zeros((N_VISUALIZE, arch['z_dim'])),
    #     name='embeddings')
    # # ======================================



    writer = tf.train.SummaryWriter(dirs['logdir'])
    writer.add_graph(tf.get_default_graph())
    
    summary_op = tf.merge_all_summaries()

    with open(os.path.join(dirs['logdir'], args.architecture), 'w') as f:
        json.dump(arch, f)

    if args.gpu_cfg:
        with open(args.gpu_cfg) as f:
            cfg = json.load(f)
        gpu_options = tf.GPUOptions(
            per_process_gpu_memory_fraction=cfg['per_process_gpu_memory_fraction'])
        session_conf = tf.ConfigProto(
            allow_soft_placement=cfg['allow_soft_placement'],
            log_device_placement=cfg['log_device_placement'],
            inter_op_parallelism_threads=cfg['inter_op_parallelism_threads'],
            intra_op_parallelism_threads=cfg['intra_op_parallelism_threads'],
            gpu_options=gpu_options)
        sess = tf.Session(
            config=session_conf)
    else:
        sess = tf.Session()

    init = tf.global_variables_initializer()
    sess.run(init)

    saver = tf.train.Saver()  # tf.global_variables()
    try:
        saved_global_step = load(saver, sess, dirs['restore_from'])
        if saved_global_step is None:
            saved_global_step = -1
    except:
        print("Something's wrong while restoing checkpoints!")
        raise

    threads = tf.train.start_queue_runners(sess=sess, coord=coord)



    # # ========== For embedding =============
    # ass_op = tf.assign(em_var, codes['mu'], name='X/em_var')

    # config = projector.ProjectorConfig()
    # embedding = config.embeddings.add()
    # embedding.tensor_name = em_var.name
    # print(em_var.name, em_var.get_shape())
    # embedding.sprite.image_path = PATH_TO_SPRITE_IMAGE
    # embedding.sprite.single_image_dim.extend([w, h])
    # embedding.metadata_path = PATH_TO_LABEL
    # projector.visualize_embeddings(writer, config)
    # # =====================================


    # ========== Actual training loop ==========
    try:
        n_iter_per_epoch = info['n_files'] // args.batch_size
        time_i = time.time()
        step = 0
        for ep in range(args.n_epoch):
            for it in range(n_iter_per_epoch):
                _, l_df, l_dr = sess.run([opt_d, loss['D_fake'], loss['D_real']])

                # Update G twice
                _, l_g = sess.run([opt_g, loss['G_fake']])
                _, l_g = sess.run([opt_g, loss['G_fake']])
                if arch['mode'] == 'VAE-GAN':
                    _, l_e, l_dis = sess.run([opt_e, loss['KL(z)'], loss['Dis']])

                # Message
                msg = 'Epoch [{:3d}/{:3d}] '.format(ep + 1, args.n_epoch)\
                    + '[{:4d}/{:4d}] '.format(it + 1, n_iter_per_epoch)\
                    + 'd_loss={:6.3f}+{:6.3f}, '.format(l_df, l_dr)\
                    + 'g_loss={:5.2f}, '.format(l_g)
                    
                if arch['mode'] == 'VAE-GAN':
                    msg += 'KLD={:6.3f}, DIS={:6.3f}, '.format(l_e, l_dis)

                msg += 'T={:.2f}'.format(time.time() - time_i)
                print(msg)

                # writer.add_summary(summary, step)

                # Demo/Output
                if it % (n_iter_per_epoch // 1) == 0:
                    summary = sess.run(summary_op)
                    writer.add_summary(summary, step)

                    if arch['mode'] == 'VAE-GAN':
                        visualize_interpolation(sess, x_interp,
                            filename=os.path.join(
                                dirs['logdir'],
                                'test-Ep{:03d}-It{:04d}.png'.format(ep, it)))
                        # sess.run(ass_op)
                    
                    visualize_random_samples(sess, xh,
                        filename=os.path.join(
                            dirs['logdir'],
                            'test-Ep{:03d}-It{:04d}-dc.png'.format(ep, it)))

                    save(saver, sess, dirs['logdir'], step)

                step += 1

    except KeyboardInterrupt:
        print()

    finally:
        save(saver, sess, dirs['logdir'], step)
        coord.request_stop()
        coord.join(threads)


if __name__ == '__main__':
    main()

