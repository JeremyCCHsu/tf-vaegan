import pdb
from tensorflow.contrib import slim
import tensorflow as tf
from util.layer import GaussianLogDensity, GaussianKLD, \
    GaussianSampleLayer, lrelu

class VAEGAN(object):
    '''
    VAE-GAN: Variational Auto-encoder with Generative Adversarial Net

    [TODO] I shall purge the legacy part of DC-GAN 
           because the API's aren't the same.
    '''
    def __init__(self, arch, is_training=False):
        self.arch = arch
        self._sanity_check()
        self.is_training = is_training

        self._generate = tf.make_template(
            'Generator',
            self._generator)
        self._discriminate = tf.make_template(
            'Discriminator',
            self._discriminator)
        self._encode = tf.make_template(
            'Encoder',
            self._encoder)


    def _sanity_check(self):
        for net in ['encoder', 'generator', 'discriminator']:
            assert len(self.arch[net]['output']) > 2
            assert len(self.arch[net]['output']) == len(self.arch[net]['kernel'])
            assert len(self.arch[net]['output']) == len(self.arch[net]['stride'])
        

    def _encoder(self, x, is_training):
        n_layer = len(self.arch['encoder']['output'])
        subnet = self.arch['encoder']
        with slim.arg_scope(
                [slim.batch_norm],
                scale=True,
                updates_collections=None,
                decay=0.9, epsilon=1e-5,
                is_training=is_training,
                reuse=None):
            with slim.arg_scope(
                    [slim.conv2d],
                    weights_regularizer=slim.l2_regularizer(subnet['l2-reg']),
                    normalizer_fn=slim.batch_norm,
                    activation_fn=lrelu):

                for i in range(n_layer):
                    x = slim.conv2d(
                        x,
                        subnet['output'][i],
                        subnet['kernel'][i],
                        subnet['stride'][i])

        x = slim.flatten(x)

        with slim.arg_scope(
            [slim.fully_connected],
            num_outputs=self.arch['z_dim'],
            weights_regularizer=slim.l2_regularizer(subnet['l2-reg']),
            normalizer_fn=None,
            activation_fn=None):
            z_mu = slim.fully_connected(x)
            z_lv = slim.fully_connected(x)
        return z_mu, z_lv


    def _generator(self, z, is_training):
        subnet = self.arch['generator']
        n_layer = len(subnet['output'])
        h, w, c = subnet['hwc']
        with slim.arg_scope(
            [slim.batch_norm],
            scale=True,
            updates_collections=None,
            decay=0.9, epsilon=1e-5,
            is_training=is_training,
            scope='BN'):

            x = slim.fully_connected(
                z,
                h * w * c,
                normalizer_fn=slim.batch_norm,
                activation_fn=tf.nn.relu)
            x = tf.reshape(x, [-1, h, w, c])

            with slim.arg_scope(
                    [slim.conv2d_transpose],
                    weights_regularizer=slim.l2_regularizer(subnet['l2-reg']),
                    normalizer_fn=slim.batch_norm,
                    activation_fn=tf.nn.relu):

                for i in range(n_layer -1):
                    x = slim.conv2d_transpose(
                        x,
                        subnet['output'][i],
                        subnet['kernel'][i],
                        subnet['stride'][i])

                # Don't apply BN for the last layer of G
                x = slim.conv2d_transpose(
                    x,
                    subnet['output'][-1],
                    subnet['kernel'][-1],
                    subnet['stride'][-1],
                    normalizer_fn=None,
                    activation_fn=tf.nn.tanh)
        return x

    def _discriminator(self, x, is_training):
        subnet = self.arch['discriminator']
        n_layer = len(subnet['output'])
        feature = list()
        with slim.arg_scope(
                [slim.batch_norm],
                scale=True,
                updates_collections=None,
                decay=0.9, epsilon=1e-5,
                is_training=is_training,
                # reuse=None
                scope='BN'):
            with slim.arg_scope(
                    [slim.conv2d],
                    weights_regularizer=slim.l2_regularizer(subnet['l2-reg']),
                    normalizer_fn=slim.batch_norm,
                    activation_fn=lrelu):

                # Radford: [do] not applying batchnorm to the discriminator input layer
                x = slim.conv2d(
                    x,
                    subnet['output'][0],
                    subnet['kernel'][0],
                    subnet['stride'][0],
                    normalizer_fn=None)
                feature.append(x)
                for i in range(1, n_layer):
                    x = slim.conv2d(
                        x,
                        subnet['output'][i],
                        subnet['kernel'][i],
                        subnet['stride'][i])
                    feature.append(x)

        # Don't apply BN for the last layer
        x = slim.flatten(x)
        h = slim.flatten(feature[subnet['feature_layer'] - 1])
        x = slim.fully_connected(
            x,
            1,
            weights_regularizer=slim.l2_regularizer(subnet['l2-reg']),
            activation_fn=None)
        return x, h  # no explicit `sigmoid`

    def loss(self, x):
        def mean_sigmoid_cross_entropy_with_logits(logit, truth):
            '''
            truth: 0. or 1.
            '''
            return tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logit,
                    truth * tf.ones_like(logit)))

        if self.arch['mode'] == 'VAE-GAN':
            z_mu, z_lv = self._encode(x, is_training=self.is_training)
            z = GaussianSampleLayer(z_mu, z_lv)

            # [Test] ================================
            z_direct = tf.random_normal(shape=tf.shape(z_mu))
            xz = self._generate(
                z_direct,
                is_training=self.is_training)
            logit_fake_xz, _ = self._discriminate(xz,
                is_training=self.is_training)
            # ========================================

        elif self.arch['mode'] == 'DC-GAN':
            batch_size = x.get_shape().as_list()[0]
            z = tf.random_uniform(
                shape=[batch_size, self.arch['z_dim']],
                minval=-1.0,
                maxval=1.0,
                name='z')
        else:
            raise ValueError('Supported mode: DC-GAN or VAE-GAN')

        xh = self._generate(z, is_training=self.is_training)

        logit_true, x_through_D = self._discriminate(x,
            is_training=self.is_training)
        logit_fake, xh_through_D = self._discriminate(xh,
            is_training=self.is_training)


        with tf.name_scope('loss'):
            loss = dict()
            loss['D_real'] = \
                mean_sigmoid_cross_entropy_with_logits(logit_true, 1.)

            loss['D_fake'] = 0.5 * (
                mean_sigmoid_cross_entropy_with_logits(logit_fake, 0.) +\
                mean_sigmoid_cross_entropy_with_logits(logit_fake_xz, 0.))
            # loss['D_fake'] /= 2.    # [TODO] Jmod

            loss['G_fake'] = 0.5 * (
                mean_sigmoid_cross_entropy_with_logits(logit_fake, 1.) +\
                mean_sigmoid_cross_entropy_with_logits(logit_fake_xz, 1.))
            # loss['G_fake'] /= 2.

            if self.arch['mode'] == "VAE-GAN":
                loss['KL(z)'] = tf.reduce_mean(
                    GaussianKLD(
                        z_mu, z_lv,
                        tf.zeros_like(z_mu), tf.zeros_like(z_lv)))

                loss['Dis'] = - tf.reduce_mean(
                    GaussianLogDensity(
                        x_through_D,
                        xh_through_D,
                        tf.zeros_like(xh_through_D)))
                
                # [Test] ================================
                loss['G_fake_xz'] = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(
                        logit_fake_xz,
                        tf.ones_like(logit_fake_xz)))
                # ========================================

            # For summaries
            with tf.name_scope('Summary'):
                tf.summary.histogram('z', z)
                tf.summary.histogram('D(true)', tf.nn.sigmoid(logit_true))
                tf.summary.histogram('D(Fake)', tf.nn.sigmoid(logit_fake))
                tf.summary.histogram('logit_true', logit_true)
                tf.summary.histogram('logit_fake', logit_fake)
                tf.summary.histogram('logit_sample', logit_fake_xz)
                tf.summary.histogram('logits',
                    tf.concat(0, [logit_fake, logit_true]))
                tf.summary.image("G", xh)
        return loss

    def sample(self, z=128):
        ''' Generate fake samples given `z`
        if z is not given or is an `int`,
        this fcn generates (z=128) samples
        '''
        # z = tf.random_uniform(
        #     shape=[z, self.arch['z_dim']],
        #     minval=-1.0,
        #     maxval=1.0,
        #     name='z_test')
        z = tf.random_normal(shape=[z, self.arch['z_dim']])
        return self._generate(z, is_training=False)

    def encode(self, x):
        z_mu, z_lv = self._encode(x, is_training=False)
        return dict(mu=z_mu, log_var=z_lv)

    def decode(self, z, y=None, tanh=False):
        return self._generate(z, is_training=False)

    def interpolate(self, x1, x2, n):
        ''' Interpolation from the latent space '''
        x1 = tf.expand_dims(x1, 0)
        x2 = tf.expand_dims(x2, 0)
        z1, _ = self._encode(x1, is_training=False)
        z2, _ = self._encode(x2, is_training=False)

        def L2norm(x):
            return tf.sqrt(tf.reduce_sum(tf.square(x), -1))

        norm1 = L2norm(z1)
        norm2 = L2norm(z2)

        theta = tf.matmul(z1/norm1, z2/norm2, transpose_b=True)

        a = tf.reshape(tf.linspace(0., 1., n), [n, 1])  # 10x1

        a1 = tf.sin((1. - a) * theta) / tf.sin(theta)
        a2 = tf.sin(a * theta) / tf.sin(theta)
        z = a1 * z1 + a2 * z2

        xh = self._generate(z, is_training=False)
        xh = tf.concat(0, [x1, xh, x2])
        return xh