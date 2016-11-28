

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

