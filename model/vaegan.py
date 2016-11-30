import pdb
from tensorflow.contrib import slim
import tensorflow as tf
import numpy as np

# [TODO] I think the upsampling used too much convs (it's OK but I probably have to try less first)

L2 = 1e-6
# L2 = 0.001
STDDEV = 0.02
EPSILON = 1e-10


def GaussianLogDensity(x, mu, log_var, name='GaussianLogDensity'):
    c = np.log(2 * np.pi)
    var = tf.exp(log_var)
    x_mu2 = tf.square(tf.sub(x, mu))   # [Issue] not sure the dim works or not?
    x_mu2_over_var = tf.div(x_mu2, var + EPSILON)
    log_prob = -0.5 * (c + log_var + x_mu2_over_var)
    log_prob = tf.reduce_sum(log_prob, -1, name=name)   # keep_dims=True,
    return log_prob


def kld_of_gaussian(mu1, log_var1, mu2, log_var2):
    ''' Kullback-Leibler divergence of two Gaussians
        *Assuming that each dimension is independent
        mu: mean
        log_var: log variance
        Equation: http://stats.stackexchange.com/questions/7440/kl-divergence-between-two-univariate-gaussians
    '''
    var = tf.exp(log_var1)
    var2 = tf.exp(log_var2)
    mu_diff_sq = tf.square(tf.sub(mu1, mu2))
    single_variable_kld = 0.5 * (log_var2 - log_var1) \
        + 0.5 * tf.div(var, var2) * (tf.add(1.0, mu_diff_sq)) - 0.5
    return tf.reduce_sum(single_variable_kld, -1)

def SampleLayer(z_mu, z_lv, name='SampleLayer'):
    with tf.name_scope(name):
        eps = tf.random_normal(tf.shape(z_mu))
        std = tf.sqrt(tf.exp(z_lv))
        return tf.add(z_mu, tf.mul(eps, std))

def lrelu(x, leak=0.2, name="lrelu"):
  return tf.maximum(x, leak*x)

class DCGAN(object):
    def __init__(self, arch, is_training=False):
        self.arch = arch
        self.is_training = is_training

        # then I have to declare x here, then the shape is determined (undesirable!)
        self._generate = tf.make_template(
            'Generator',
            self._generator)
        self._discriminate = tf.make_template(
            'Discriminator',
            self._discriminator)
        self._encode = tf.make_template(
            'Encoder',
            self.encoder)


    def encoder(self, x, is_training):
        with slim.arg_scope(
                [slim.batch_norm],
                scale=True,
                # updates_collections=None,
                is_training=is_training,
                reuse=None):
            with slim.arg_scope(
                    [slim.conv2d],
                    kernel_size=[5, 5],
                    stride=[2, 2],
                    normalizer_fn=slim.batch_norm,
                    activation_fn=lrelu):
                for i in range(4):
                    x = slim.conv2d(x, self.arch['ch_D'] * (2 ** i))
                x = slim.flatten(x)
                # print(x, x.get_shape())
                z_mu = slim.fully_connected(x, self.arch['z_dim'],
                    normalizer_fn=None,
                    activation_fn=None)
                z_lv = slim.fully_connected(x, self.arch['z_dim'],
                    normalizer_fn=None,
                    activation_fn=None)
        return z_mu, z_lv



    def _generator(self, z, is_training):
        # Fixed upsampling rate
        w = int(self.arch['img_w'] // 16)
        h = int(self.arch['img_h'] // 16)
        c = self.arch['img_c']
        ch = self.arch['ch_D']

        if is_training:
            reuse = None
        else:
            reuse = True

        with slim.arg_scope(
                [slim.batch_norm],
                scale=True,
                # updates_collections=None,
                is_training=is_training,
                reuse=reuse):
            with slim.arg_scope(
                    [slim.conv2d_transpose],
                    kernel_size=[5, 5],
                    stride=[2, 2],
                    # weights_regularizer=slim.l2_regularizer(L2),
                    # normalizer_fn=slim.batch_norm,
                    # activation_fn=tf.nn.relu
                    normalizer_fn=None,
                    activation_fn=None):

                # x = slim.fully_connected(z, h * w * ch * 8,
                #     normalizer_fn=slim.batch_norm,
                #     activation_fn=tf.nn.relu,
                #     scope='BN-8')
                x = slim.fully_connected(z, h * w * ch * 8,
                    activation_fn=None)
                x = slim.batch_norm(x, scope='BN-8')
                x = tf.nn.relu(x)

                x = tf.reshape(x, [-1, h, w, ch * 8])
                for i in [4, 2, 1]:
                    x = slim.conv2d_transpose(x, ch * i)
                    x = slim.batch_norm(x, scope='BN-{:d}'.format(i))
                    x = tf.nn.relu(x)

                # Don't apply BN for the last layer of G
                x = slim.conv2d_transpose(x, c,
                    normalizer_fn=None,
                    activation_fn=tf.nn.tanh)
        return x

    def _discriminator(self, x, is_training):
        ch = self.arch['ch_D']

        with slim.arg_scope(
                [slim.batch_norm],
                scale=True,
                # updates_collections=None,
                is_training=is_training,
                reuse=None):
            with slim.arg_scope(
                    [slim.conv2d],
                    kernel_size=[5, 5],
                    stride=[2, 2],
                    # weights_regularizer=slim.l2_regularizer(L2),
                    # normalizer_fn=slim.batch_norm,
                    normalizer_fn=None,
                    activation_fn=None):

                # Radford: not applying batchnorm to the discriminator input layer
                x = slim.conv2d(x, ch)
                # =========== [TEST] =============
                # J: This seemed to be harmless
                x = slim.batch_norm(x, scope='Bn-1')
                # ================================
                x = lrelu(x)
                for i in [2, 4, 8]:
                    x = slim.conv2d(x, ch * i)
                    x = slim.batch_norm(x, scope='Bn-{:d}'.format(i))
                    x = lrelu(x)
        
        # Don't apply BN for the last layer
        x = slim.flatten(x)
        h = x
        x = slim.fully_connected(x, 1,
            # weights_regularizer=slim.l2_regularizer(L2)
            activation_fn=None)
        return x, h  # no explicit `sigmoid`

    def loss(self, x):
        batch_size = x.get_shape().as_list()[0]
        # [TODO] Maybe I should make sampling stratified
        # z = tf.random_uniform(
        #     shape=[batch_size, self.arch['z_dim']],
        #     minval=-1.,
        #     maxval=1.,
        #     name='z')
        # z_mu = z
        # z_lv = z

        z_mu, z_lv = self._encode(x, is_training=self.is_training)
        z = SampleLayer(z_mu, z_lv)
        # print(z_mu.get_shape(), z_lv.get_shape())

        xh = self._generate(z, is_training=self.is_training)
        self.xh = xh

        # pdb.set_trace()

        logit_fake, xh_through_D = self._discriminate(xh, is_training=self.is_training)
        logit_true, x_through_D = self._discriminate(x, is_training=self.is_training)

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

            loss['KL(z)'] = tf.reduce_mean(
                kld_of_gaussian(
                    z_mu, z_lv,
                    tf.zeros_like(z_mu), tf.zeros_like(z_lv)))

            loss['Dis'] = - tf.reduce_mean(
                GaussianLogDensity(
                    x_through_D,
                    xh_through_D,
                    tf.zeros_like(xh_through_D)))

        return loss

    def sample(self, z=128):
        ''' Generate fake samples given `z`
        if z is not given or is an `int`,
        this fcn generates (z=128) samples
        '''
        # z = tf.random_uniform(
        #     shape=[z, self.arch['z_dim']],
        #     minval=-.5,
        #     maxval=.5,
        #     name='z_test')
        # return self.generate(z, is_training=False)
        return self.xh  # [BUG] called before assigned
        # return xh


    def encode(self, x):
        z_mu, z_lv = self._encode(x, is_training=False)
        return dict(mu=z_mu, log_var=z_lv)

    def decode(self, z, y=None):
        return self._generate(z, is_training=False)
        
    def interpolate(self, x1, x2, n):
        ''' Interpolation from the latent space '''
        # z's should be 1x100
        z1, _ = self._encode(
            tf.expand_dims(x1, 0),
            is_training=False)
        z2, _ = self._encode(
            tf.expand_dims(x2, 0),
            is_training=False)
        a = tf.reshape(tf.linspace(0., 1., n), [n, 1])

        z1 = tf.matmul(1. - a, z1)
        z2 = tf.matmul(a, z2)
        z = tf.add(z1, z2)
        # z = tf.add(z1, tf.matmul(a, z2))
        xh = self._generate(z, is_training=False)
        # xh = tf.transpose(xh, [1, 2, 3, 0])
        # xh = tf.reshape()
        return xh