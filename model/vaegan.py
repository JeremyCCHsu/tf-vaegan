from tensorflow.contrib import slim
import tensorflow as tf

# [TODO] I think the upsampling used too much convs (it's OK but I probably have to try less first)

L2 = 1e-6



class batch_norm(object):
    """Code modification of http://stackoverflow.com/a/33950177"""
    def __init__(self, epsilon=1e-5, momentum = 0.9, name="batch_norm"):
        with tf.variable_scope(name):
            self.epsilon = epsilon
            self.momentum = momentum

            self.ema = tf.train.ExponentialMovingAverage(decay=self.momentum)
            self.name = name

    def __call__(self, x, train=True):
        shape = x.get_shape().as_list()

        if train:
            with tf.variable_scope(self.name) as scope:
                self.beta = tf.get_variable("beta", [shape[-1]],
                                    initializer=tf.constant_initializer(0.))
                self.gamma = tf.get_variable("gamma", [shape[-1]],
                                    initializer=tf.random_normal_initializer(1., 0.02))
                
                try:
                    batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], name='moments')
                except:
                    batch_mean, batch_var = tf.nn.moments(x, [0, 1], name='moments')
                    
                ema_apply_op = self.ema.apply([batch_mean, batch_var])
                self.ema_mean, self.ema_var = self.ema.average(batch_mean), self.ema.average(batch_var)

                with tf.control_dependencies([ema_apply_op]):
                    mean, var = tf.identity(batch_mean), tf.identity(batch_var)
        else:
            mean, var = self.ema_mean, self.ema_var

        normed = tf.nn.batch_norm_with_global_normalization(
                x, mean, var, self.beta, self.gamma, self.epsilon, scale_after_normalization=True)

        return normed


def ConvDsmpLayer(
    x,
    ch_out,
    f_size=[5, 5],
    stride=[2, 2], L2=L2,
    is_training=False):
    with slim.arg_scope(
            [slim.conv2d],
            activation_fn=tf.nn.relu,
            stride=[1, 1],
            weights_regularizer=slim.l2_regularizer(L2)):
        # x = slim.conv2d(x, ch_out, f_size)
        x = slim.conv2d(x, ch_out, f_size, stride=stride)

        # [TODO] conv (lin) > batch norm > relu
        x = slim.batch_norm(x, is_training=is_training)
    return x


# def ConvDsmpLayer()

def UsmpConvLayer(x, ch_out, u_size=[5, 5], f_size=[3, 3], stride=[2, 2],
        activation_fn=tf.nn.relu,
        L2=L2,
        is_training=False,
        bn=True):
    with slim.arg_scope(
            [slim.conv2d_transpose],
            activation_fn=activation_fn,
            stride=[1, 1],
            weights_regularizer=slim.l2_regularizer(L2)):
        x = slim.conv2d_transpose(x, ch_out, u_size, stride=stride)

        if bn:
            x = slim.batch_norm(x, is_training=is_training)
        # x = slim.conv2d(x, ch_out, f_size,
        #   activation_fn=tf.nn.relu,
        #   weights_regularizer=slim.l2_regularizer(L2))
    return x

def FullConnLayer(
        x,
        out,
        activation_fn=tf.nn.relu,
        is_training=False,
        bn=True):
    with slim.arg_scope(
            [slim.fully_connected],
            activation_fn=activation_fn,
            weights_regularizer=slim.l2_regularizer(L2)):
        if bn:
            x = slim.batch_norm(x, is_training=is_training)
        x = slim.fully_connected(x, out)
    return x
#       x = slim.fully_connected(x, out)


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
        # Fixe upsampling rate
        w = int(self.arch['img_w'] // 16)
        h = int(self.arch['img_h'] // 16)
        c = self.arch['img_c']
        ch = self.arch['ch_D']
        x = FullConnLayer(z, w*h*ch*8, bn=True, is_training=is_training)
        x = tf.reshape(x, [-1, h, w, ch*8])
        x = slim.batch_norm(x, is_training=is_training)

        # x = UsmpConvLayer(x, ch*4, u_size=[5, 5], f_size=[3, 3],
        #     is_training=is_training)
        # x = UsmpConvLayer(x, ch*2, u_size=[5, 5], f_size=[3, 3],
        #     is_training=is_training)
        # x = UsmpConvLayer(x, ch*1, u_size=[5, 5], f_size=[3, 3],
        #     is_training=is_training)
        # x = UsmpConvLayer(x, c, u_size=[5, 5], f_size=[3, 3],
        #     is_training=is_training,
        #     bn=False,
        #     activation_fn=tf.nn.sigmoid)

        with slim.arg_scope(
                [slim.conv2d_transpose],
                activation_fn=tf.nn.relu,
                kernel_size=[5, 5],
                stride=[2, 2],
                weights_regularizer=slim.l2_regularizer(L2)):
            for i in [4, 2, 1]:
                x = slim.conv2d_transpose(x, ch * i)
                # bn = batch_norm(name='bn_g_{:d}'.format(i))
                # x = bn(x, train=is_training)
                x = slim.batch_norm(x, is_training=is_training)

            x = slim.conv2d_transpose(x, c,
                activation_fn=tf.nn.sigmoid)

        return x

    def discriminator(self, x, is_training):
        ch = self.arch['ch_D']
        # x = ConvDsmpLayer(x, ch*1, is_training=is_training)
        # x = ConvDsmpLayer(x, ch*2, is_training=is_training)
        # x = ConvDsmpLayer(x, ch*4, is_training=is_training)
        # x = ConvDsmpLayer(x, ch*8, is_training=is_training)

        with slim.arg_scope(
                [slim.conv2d],
                # activation_fn=tf.nn.relu,
                activation_fn=lrelu,
                kernel_size=[5, 5],
                stride=[2, 2],
                weights_regularizer=slim.l2_regularizer(L2)):
            for i in [1, 2, 4, 8]:
                x = slim.conv2d(x, ch*i)
                bn = batch_norm(name='bn{:d}'.format(i))
                x = bn(x, train=is_training)

        # total_dim = tf.reduce_prod(x.get_shape()[1:])
        # total_dim = 1
        # for i in x.get_shape().as_list()[1:]:
        #     total_dim *= i
        # x = tf.reshape(x, [-1, total_dim])
        x = slim.flatten(x)

        # x = FullConnLayer(x, 1,
        #     activation_fn=tf.identity,
        #     bn=False,
        #     is_training=is_training)  # logit
        
        x = slim.fully_connected(x, 1,
            activation_fn=tf.identity,
            weights_regularizer=slim.l2_regularizer(L2))

        return x

    def loss(self, x):
        batch_size = x.get_shape().as_list()[0]
        # [TODO] Maybe I should make sampling stratified
        z = tf.random_uniform(
            shape=[batch_size, self.arch['z_dim']],
            minval=0.,
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
            minval=0.,
            maxval=1.,
            name='z')
        return self.generate(z, is_training=False)
        # return xh








# How to test?
# Test on a data set

