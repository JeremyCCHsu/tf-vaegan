import os
import sys
import time
import json
import tensorflow as tf

from datetime import datetime

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
# n_step = 16300
batch_size = 64

STARTED_DATESTRING = datetime.now().strftime('%0m%0d-%0H%0M-%0S-%Y')
N_INTERP = 10
LOGDIR_ROOT = './logdir'

args = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string(
    'datadir', '/home/jrm/proj/Hanzi/TWKai_98_32x32', 'dir to dataset')
tf.app.flags.DEFINE_string(
    'logdir_root', None, 'root of log dir')
tf.app.flags.DEFINE_string(
    'logdir', None, 'log dir')
tf.app.flags.DEFINE_string(
    'restore_from', None, 'restore from dir (not from *.ckpt)')
tf.app.flags.DEFINE_integer('n_epoch', 100, 'num of epoch')
tf.app.flags.DEFINE_integer('batch_size', 64, 'batch size')
tf.app.flags.DEFINE_float('lr', 2e-4, 'learning rate')
tf.app.flags.DEFINE_float('beta1', 0.5, 'beta1 in AdamOptimizer')
tf.app.flags.DEFINE_boolean('train', False, 'is training or not')

# [TODO] should I put this in a util dir?
def save(saver, sess, logdir, step):
    ''' Save a model to logdir/model.ckpt-[step] '''
    model_name = 'model.ckpt'
    checkpoint_path = os.path.join(logdir, model_name)
    print('Storing checkpoint to {} ...'.format(logdir), end="")
    sys.stdout.flush()

    if not os.path.exists(logdir):
        os.makedirs(logdir)

    saver.save(sess, checkpoint_path, global_step=step)
    print(' Done.')


def load(saver, sess, logdir):
    '''
    Try to load model form a dir (search for the newest checkpoint)
    '''
    print('Trying to restore checkpoints from {} ...'.format(logdir),
        end="")
    ckpt = tf.train.get_checkpoint_state(logdir)
    if ckpt:
        print('  Checkpoint found: {}'.format(ckpt.model_checkpoint_path))
        global_step = int(
            ckpt.model_checkpoint_path
            .split('/')[-1]
            .split('-')[-1])
        print('  Global step: {}'.format(global_step))
        print('  Restoring...', end="")
        saver.restore(sess, ckpt.model_checkpoint_path)
        return global_step
    else:
        print('No checkpoint found')
        return None


def visualize_random_samples(sess, imgs, xh, filename=None):
    img_real = sess.run(imgs)
    img_fake = sess.run(xh)
    
    plt.figure()
    for i in range(15):
        plt.subplot(4, 4, i + 1)
        plt.imshow(img_fake[i, :, :, 0], interpolation='none', cmap='gray')
        plt.colorbar()
    plt.subplot(4, 4, 16)
    plt.imshow(img_real[0, :, :, 0], interpolation='none', cmap='gray')
    plt.colorbar()

    if filename:
        plt.savefig(filename)
        plt.close()


def visualize_interpolation(sess, x_interp, filename=None):
    x_ = sess.run(x_interp)
    plt.figure(figsize=(x_.shape[0], 1))
    for i in range(x_.shape[0]):
        plt.subplot(1, x_.shape[0], i + 1)
        plt.imshow(x_[i, :, :, 0], interpolation='none', cmap='gray')
        plt.axis('off')
        # plt.colorbar()
    
    if filename:
        plt.savefig(filename)
        plt.close()


def get_optimization_ops(loss):
    trainables = tf.trainable_variables()
    g_vars = [v for v in trainables if 'Generator' in v.name]
    d_vars = [v for v in trainables if 'Discriminator' in v.name]
    e_vars = [v for v in trainables if 'Encoder' in v.name]

    optimizer = tf.train.AdamOptimizer(args.lr, args.beta1)

    # # Vanila GAN
    # optimizer = tf.train.AdamOptimizer(args.lr)
    # obj_D = loss['D_fake'] + loss['D_real']
    # obj_G = loss['G_fake']
    # optimizer = tf.train.RMSPropOptimizer(args.lr)
    # opt_d = optimizer.minimize(
    #   obj_D,
    #   var_list=d_vars)
    # opt_g = optimizer.minimize(
    #   loss['G_fake'],
    #   var_list=g_vars)

    # VAE-GAN
    obj_D = loss['D_fake'] + loss['D_real']
    obj_G = loss['Dis'] + loss['G_fake']
    obj_E = loss['KL(z)'] + loss['Dis']

    opt_d = optimizer.minimize(obj_D, var_list=d_vars)
    opt_g = optimizer.minimize(obj_G, var_list=g_vars)
    opt_e = optimizer.minimize(obj_E, var_list=e_vars)
    return opt_d, opt_g, opt_e


def get_default_logdir(logdir_root):
    return  os.path.join(logdir_root, 'train', STARTED_DATESTRING)

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
    arch = json.load(open('architecture.json'))
    imgs, info = img_reader(
        datadir=datadir,
        img_dims=(arch['img_h'], arch['img_w'], arch['img_c']),
        batch_size=batch_size,
        rtype='tanh')
    
    machine = DCGAN(arch, is_training=True)

    loss = machine.loss(imgs)
    xh = machine.sample(batch_size)
    
    x_interp = machine.interpolate(imgs[0], imgs[1], N_INTERP)

    opt_d, opt_g, opt_e = get_optimization_ops(loss)

    writer = tf.train.SummaryWriter(dirs['logdir'])
    writer.add_graph(tf.get_default_graph())

    sess = tf.Session()
    init = tf.initialize_all_variables()
    sess.run(init)

    saver = tf.train.Saver()
    try:
        saved_global_step = load(saver, sess, dirs['restore_from'])
        if saved_global_step is None:
            saved_global_step = -1
    except:
        print("Something's wrong while restoing checkpoints!")
        raise

    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    # ========== Actual training loop ==========
    try:
        n_iter_per_epoch = info['n_files'] // batch_size
        time_i = time.time()
        for ep in range(n_epoch):
            for it in range(n_iter_per_epoch):
                _, l_df, l_dr = sess.run([opt_d, loss['D_fake'], loss['D_real']])
                _, l_g = sess.run([opt_g, loss['G_fake']])
                _, l_e, l_dis = sess.run([opt_e, loss['KL(z)'], loss['Dis']])

                t = time.time() - time_i

                print(
                    'Epoch [{:3d}/{:3d}] '.format(ep, n_epoch) +
                    '[{:4d}/{:4d}] '.format(it, n_iter_per_epoch) +
                    'd_loss={:6.3f}+{:6.3f}, '.format(l_df, l_dr) +
                    'g_loss={:5.2f}, T={:.2f}'.format(l_g, t)
                    )

                if it % (n_iter_per_epoch // 2) == 0:   
                    # visualize_random_samples(sess, imgs, xh,
                    #   filename='test-Ep{:03d}-It{:04d}.png'.format(ep, it))
                    visualize_interpolation(sess, x_interp,
                        filename=os.path.join(
                            dirs['logdir'],
                            'test-Ep{:03d}-It{:04d}.png'.format(ep, it)))

    except KeyboardInterrupt:
        print()

    finally:
        save(saver, sess, dirs['logdir'], ep * n_iter_per_epoch + it)
        coord.request_stop()
        coord.join(threads)

if __name__ == '__main__':
    main()

