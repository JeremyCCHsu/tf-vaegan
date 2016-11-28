from tensorflow.contrib import slim
import tensorflow as tf

# [TODO] I think the upsampling used too much convs (it's OK but I probably have to try less first)

L2 = 1e-6
# L2 = 0.001
STDDEV = 0.02

def lrelu(x, leak=0.2, name="lrelu"):
  return tf.maximum(x, leak*x)

class DCGAN(object):
    def __init__(self, arch, is_training=False):
        self.arch = arch
        self.is_training = is_training

        # then I have to declare x here, then the shape is determined (undesirable!)
        self.generate = tf.make_template(
            'Generator',
            self.generator)
        self.discriminate = tf.make_template(
            'Discriminator',
            self.discriminator)

    def generator(self, z, is_training):
        # Fixed upsampling rate
        w = int(self.arch['img_w'] // 16)
        h = int(self.arch['img_h'] // 16)
        c = self.arch['img_c']
        ch = self.arch['ch_D']

        x = slim.fully_connected(z,
            h * w * ch * 8,
            activation_fn=tf.nn.relu,
            # weights_initializer=tf.random_normal_initializer(stddev=STDDEV),
            weights_regularizer=slim.l2_regularizer(L2))
        x = tf.reshape(x, [-1, h, w, ch * 8])
        x = slim.batch_norm(x, scale=True, is_training=is_training)

        with slim.arg_scope(
                [slim.conv2d_transpose],
                activation_fn=tf.nn.relu,
                kernel_size=[5, 5],
                stride=[2, 2],
                # weights_initializer=tf.random_normal_initializer(stddev=STDDEV),
                weights_regularizer=slim.l2_regularizer(L2)):
            for i in [4, 2, 1]:
                x = slim.conv2d_transpose(x, ch * i)
                x = slim.batch_norm(x, scale=True, is_training=is_training)

            x = slim.conv2d_transpose(x, c,
                activation_fn=tf.nn.tanh)

        return x

    def discriminator(self, x, is_training):
        ch = self.arch['ch_D']
        with slim.arg_scope(
                [slim.conv2d],
                activation_fn=lrelu,
                kernel_size=[5, 5],
                stride=[2, 2],
                # weights_initializer=tf.random_normal_initializer(stddev=STDDEV),
                weights_regularizer=slim.l2_regularizer(L2)):
            for i in [1, 2, 4, 8]:
                x = slim.conv2d(x, ch*i)
                x = slim.batch_norm(x, scale=True, is_training=is_training)

        x = slim.flatten(x)

        x = slim.fully_connected(x, 1,
            activation_fn=tf.identity,
            # weights_initializer=tf.random_normal_initializer(stddev=STDDEV)
            weights_regularizer=slim.l2_regularizer(L2))
        return x

    def loss(self, x):
        batch_size = x.get_shape().as_list()[0]
        # [TODO] Maybe I should make sampling stratified
        z = tf.random_uniform(
            shape=[batch_size, self.arch['z_dim']],
            minval=-1.,
            maxval=1.,
            name='z')
        xh = self.generate(z, is_training=self.is_training)
        logit_fake = self.discriminate(xh, is_training=self.is_training)
        logit_true = self.discriminate(x, is_training=self.is_training)

        with tf.name_scope('loss'):
            loss = dict()
            loss['D_real'] = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logit_true,
                    tf.ones_like(logit_true)))

            loss['D_fake'] = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logit_fake,
                    tf.zeros_like(logit_fake)))

            loss['G_fake'] = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logit_fake,
                    tf.ones_like(logit_fake)))

        return loss

    def decode(self, z=128):
        ''' Generate fake samples given `z`
        if z is not given or is an `int`,
        this fcn generates (z=128) samples
        '''
        z = tf.random_uniform(
            shape=[z, self.arch['z_dim']],
            minval=-.5,
            maxval=.5,
            name='z_test')
        return self.generate(z, is_training=False)
        # return xh

