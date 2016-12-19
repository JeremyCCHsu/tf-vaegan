import os
import sys
import tensorflow as tf

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
