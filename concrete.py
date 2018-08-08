import tensorflow as tf
import numpy as np
from tensorflow.contrib.distributions import (ExpRelaxedOneHotCategorical,
                                              MultivariateNormalTriL,
                                              fill_triangular)
from tensorflow.contrib.keras import layers, models


def get_model_params(n_states, obs_dim, n_layers, layer_dim,
                     temp_gen, temp_rec):
    """Returns a dictionary of all model parameters

    Args:
        n_states: Number of states modeled.
        obs_dim: Dimension of observations.
        n_layers: Number of dense layers in the neural network.
        layer_dim: Number of hidden units in each dense layer.
        temp_gen: Temperature of concrete distributions in generative model.
        temp_rec: Temperature of concrete distributions in recognition model.
    """

    with tf.variable_scope("model_params"):
        with tf.variable_scope("generative"):
            gen_alpha_0 = tf.get_variable(
                "init_logits", [n_states], tf.float32,
                tf.random_normal_initializer)
            gen_A = tf.get_variable(
                "trans_matrix", [n_states, n_states], tf.float32,
                tf.random_normal_initializer)
            loc = tf.get_variable(
                "loc", [n_states, obs_dim], tf.float32,
                tf.random_normal_initializer)
            scale_tril = tf.get_variable(
                "scale_tril",
                [n_states, obs_dim * (obs_dim + 1) / 2], tf.float32,
                tf.random_normal_initializer)
        with tf.variable_scope("recognition"):
            rec_alpha_0 = tf.get_variable(
                "init_logits", [n_states], tf.float32,
                tf.random_normal_initializer)
            NN = models.Sequential(name="trans_matrix_NN")
            NN.add(layers.Dense(layer_dim, input_dim=obs_dim * 2,
                                activation="relu", name="layer_1"))
            if n_layers > 2:
                for i in range(2, n_layers):
                    NN.add(layers.Dense(layer_dim, activation="relu",
                                        name="layer_{}".format(i)))
            NN.add(layers.Dense(n_states ** 2, activation="linear",
                                name="layer_{}".format(n_layers)))

        params = dict(
            K=n_states, D=obs_dim, t_gen=temp_gen, t_rec=temp_rec,
            gen_alpha_0=gen_alpha_0, gen_A=gen_A,
            loc=loc, scale_tril=scale_tril,
            rec_alpha_0=rec_alpha_0, NN=NN)

    return params


class HMM_gen(object):
    """Generative model for HMM
    """

    def __init__(self, observations, parameters, *args, **kwargs):
        name = kwargs.get("name", "generative")
        with tf.name_scope(name):
            self.obs = tf.identity(observations, "observations")
            self.K = parameters["K"]
            self.D = parameters["D"]
            self.temp = tf.identity(parameters["t_gen"], "temperature")

            self.alpha_0 = tf.identity(parameters["gen_alpha_0"],
                                       "initial_state_logits")
            self.prob_0 = tf.nn.softmax(
                self.alpha_0, name="initial_state_probability")
            self.log_init = ExpRelaxedOneHotCategorical(
                self.temp, logits=self.alpha_0, name="initial_distribution")
            self.trans = tf.identity(parameters["gen_A"], "transition_matrix")

            self.loc = tf.identity(parameters["loc"], "loc")
            # cholesky factor of covariance matrix
            self.scale_tril = fill_triangular(
                parameters["scale_tril"], name="scale_tril")
            # self.scale_tril = tf.add(
            #     fill_triangular(parameters["scale_tril"]),
            #     1e-6 * tf.eye(self.D, batch_shape=[self.K]), "scale_tril")
            self.cov = tf.matmul(
                self.scale_tril, tf.matrix_transpose(self.scale_tril),
                name="cov")
            self.em = MultivariateNormalTriL(
                self.loc, self.scale_tril, name="emission")

            self.var = [parameters["gen_alpha_0"], parameters["gen_A"],
                        parameters["loc"], parameters["scale_tril"]]

        super(HMM_gen, self).__init__(*args, **kwargs)

    def log_prob(self, value):
        log_prob_0 = tf.add(
            tf.reduce_logsumexp(self.em.log_prob(self.obs[0]) + value[0]),
            self.log_init.log_prob(value[0]))

        log_prob = tf.reduce_sum(tf.map_fn(
            lambda x: tf.add(
                tf.reduce_logsumexp(self.em.log_prob(x[0]) + x[1]),
                ExpRelaxedOneHotCategorical(
                    self.temp, logits=tf.reshape(
                        tf.matmul(self.trans, tf.reshape(x[2], [self.K, 1])),
                        [self.K])).log_prob(x[1])),
            (self.obs[1:], value[1:], value[:-1]), tf.float32))

        return (log_prob_0 + log_prob)

    def sample(self, seed=None):
        init = self.log_init.sample()
        samp = tf.scan(lambda a, _: ExpRelaxedOneHotCategorical(
            self.temp, logits=tf.reshape(
                tf.matmul(self.trans, tf.reshape(a, [self.K, 1])),
                [self.K])).sample(), self.obs[:-1], init)

        return tf.concat([tf.reshape(init, [1, self.K]), samp], 0)


class HMM_rec(object):
    """Recognition model for HMM
    """

    def __init__(self, observations, parameters, *args, **kwargs):
        name = kwargs.get("name", "recognition")
        with tf.name_scope(name):
            self.obs = tf.identity(observations, "observations")
            self.K = parameters["K"]
            self.D = parameters["D"]
            self.temp = tf.identity(parameters["t_rec"], "temperature")

            self.alpha_0 = tf.identity(parameters["rec_alpha_0"],
                                       "initial_state_logits")
            self.log_init = ExpRelaxedOneHotCategorical(
                self.temp, logits=self.alpha_0, name="initial_distribution")

            self.NN = parameters["NN"]
            self.NN_inputs = tf.concat(
                [self.obs[:-1], self.obs[1:]], -1, "NN_inputs")
            self.trans = tf.reshape(
                self.NN(self.NN_inputs), [-1, self.K, self.K],
                "transition_matrix")

            self.var = [parameters["rec_alpha_0"]] + self.NN.variables

        super(HMM_rec, self).__init__(*args, **kwargs)

    def log_prob(self, value):
        value = tf.reshape(value, [-1, self.K])
        log_prob_0 = self.log_init.log_prob(value[0])

        log_prob = tf.reduce_sum(tf.map_fn(
            lambda x: ExpRelaxedOneHotCategorical(
                self.temp,
                logits=tf.reshape(
                    tf.matmul(x[0], tf.reshape(x[1], [self.K, 1])),
                    [self.K])).log_prob(x[2]),
            (self.trans, value[:-1], value[1:]), tf.float32))

        return (log_prob_0 + log_prob)

    def sample(self, seed=None):
        init = self.log_init.sample()
        samp = tf.scan(lambda a, x: ExpRelaxedOneHotCategorical(
            self.temp, logits=tf.reshape(
                tf.matmul(x, tf.reshape(a, [self.K, 1])),
                [self.K])).sample(), self.trans, init)

        return tf.concat([tf.reshape(init, [1, self.K]), samp], 0)
