import numpy as np
import tensorflow as tf

EPSILON = 1e-6

def GaussianLogDensity(x, mu, log_var, name='GaussianLogDensity'):
    c = np.log(2 * np.pi)
    var = tf.exp(log_var)
    x_mu2 = tf.square(tf.sub(x, mu))   # [Issue] not sure the dim works or not?
    x_mu2_over_var = tf.div(x_mu2, var + EPSILON)
    log_prob = -0.5 * (c + log_var + x_mu2_over_var)
    log_prob = tf.reduce_sum(log_prob, -1, name=name)   # keep_dims=True,
    return log_prob


def GaussianKLD(mu1, log_var1, mu2, log_var2):
    ''' Kullback-Leibler divergence of two Gaussians
        *Assuming that each dimension is independent
        mu: mean
        log_var: log variance
        Equation: http://stats.stackexchange.com/questions/7440/kl-divergence-between-two-univariate-gaussians
    '''
    with tf.name_scope('GaussianKLD'):
        var = tf.exp(log_var1)
        var2 = tf.exp(log_var2)
        mu_diff_sq = tf.square(tf.sub(mu1, mu2))
        single_variable_kld = 0.5 * (log_var2 - log_var1) \
            + 0.5 * tf.div(var, var2) * (tf.add(1.0, mu_diff_sq)) - 0.5
        return tf.reduce_sum(single_variable_kld, -1)


def GaussianSampleLayer(z_mu, z_lv, name='GaussianSampleLayer'):
    with tf.name_scope(name):
        eps = tf.random_normal(tf.shape(z_mu))
        std = tf.sqrt(tf.exp(z_lv))
        return tf.add(z_mu, tf.mul(eps, std))


def lrelu(x, leak=0.02, name="lrelu"):
    ''' Leaky ReLU '''
    return tf.maximum(x, leak*x)
