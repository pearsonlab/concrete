import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow_probability import distributions as tfd


def get_HMM_params(num_states, obs_dim, gen_layer_sizes, rec_layer_sizes,
                   t_gen, t_rec):
    """Returns a dictionary of all HMM model parameters
    """

    with tf.variable_scope("model_params"):
        with tf.variable_scope("generative"):
            gen_init_state = tf.get_variable(
                "init_state", [num_states], tf.float32,
                tf.random_normal_initializer)

            gen_A = models.Sequential(name="trans_NN")
            gen_A.add(layers.InputLayer((num_states,)))
            for num_hidden_units in gen_layer_sizes:
                gen_A.add(layers.Dense(num_hidden_units, "relu"))
            gen_A.add(layers.Dense(num_states, "linear"))

            loc = tf.get_variable(
                "loc", [num_states, obs_dim], tf.float32,
                tf.random_normal_initializer)
            scale_tril = tf.get_variable(
                "scale_tril", [num_states, obs_dim * (obs_dim + 1) / 2],
                tf.float32, tf.random_normal_initializer)

        with tf.variable_scope("recognition"):
            rec_init_state = tf.get_variable(
                "init_state", [num_states], tf.float32,
                tf.random_normal_initializer)

            rec_A = models.Sequential(name="trans_NN")
            rec_A.add(layers.InputLayer((num_states + obs_dim * 2,)))
            for num_hidden_units in rec_layer_sizes:
                rec_A.add(layers.Dense(num_hidden_units, "relu"))
            rec_A.add(layers.Dense(num_states, "linear"))

        params = dict(
            K=num_states, D=obs_dim, t_gen=t_gen, t_rec=t_rec,
            gen_init_state=gen_init_state, gen_A=gen_A,
            loc=loc, scale_tril=scale_tril,
            rec_init_state=rec_init_state, rec_A=rec_A)

    return params


class HMM_gen(object):
    """Generative model for HMM
    """

    def __init__(self, parameters, name="generative"):
        with tf.name_scope(name):
            self.K = parameters["K"]
            self.D = parameters["D"]
            self.temp = tf.identity(parameters["t_gen"], "temperature")

            self.init_state = tf.nn.log_softmax(parameters["gen_init_state"],
                                                name="initial_state")
            self.init_prob = tf.nn.softmax(parameters["gen_init_state"],
                                           name="initial_state_probability")
            self.trans = parameters["gen_A"]

            self.loc = tf.identity(parameters["loc"], "loc")
            # cholesky factor of covariance matrix
            self.scale_tril = tfd.fill_triangular(
                parameters["scale_tril"], name="scale_tril")
            # self.scale_tril = tf.add(
            #     fill_triangular(parameters["scale_tril"]),
            #     1e-6 * tf.eye(self.D, batch_shape=[self.K]), "scale_tril")
            self.cov = tf.matmul(
                self.scale_tril, tf.matrix_transpose(self.scale_tril),
                name="cov")
            self.em = tfd.MultivariateNormalTriL(
                self.loc, self.scale_tril, name="emission")

            self.var = [parameters["gen_init_state"],
                        *parameters["gen_A"].trainable_variables,
                        parameters["loc"], parameters["scale_tril"]]

    def log_likelihood_data(self, obs, states):
        return tf.reduce_sum(tf.reduce_logsumexp(self.em.log_prob(tf.tile(
            tf.reshape(obs, [-1, 1, self.D]), [1, self.K, 1])) + states, -1))

    def log_prob_states(self, states):
        # return tf.subtract(
        #     tf.reduce_sum(tfd.ExpRelaxedOneHotCategorical(
        #         self.temp, logits=self.trans(states[:-1])).log_prob(
        #         states[1:])),
        #     .1 * tf.reduce_sum(tf.abs(self.trans(states[:-1]))) / self.K)
        return tf.reduce_sum(tfd.ExpRelaxedOneHotCategorical(
            self.temp, logits=self.trans(states[:-1])).log_prob(
            states[1:]))

    def _update(self, a, _):
        logits = tf.reshape(
            self.trans(tf.reshape(a[0], [1, self.K])), [self.K])
        sample = tfd.ExpRelaxedOneHotCategorical(
            self.temp, logits=logits).sample()

        return (sample, logits)

    def sample_states(self, T):
        samples, logits = tf.scan(
            self._update, tf.zeros([T - 1]),
            (self.init_state, tf.zeros([self.K], tf.float32)))

        return tf.concat([tf.reshape(self.init_state, [1, self.K]),
                          samples], 0), logits

    def sample_data(self, T, probs=None):
        if probs is None:
            probs = tf.exp(self.sample_states(T)[0])
        em = self.em.sample(T)

        return tf.reduce_sum(tf.expand_dims(probs, -1) * em, 1)


class HMM_rec(object):
    """Recognition model for HMM
    """

    def __init__(self, observations, parameters, name = "recognition"):
        with tf.name_scope(name):
            self.obs = tf.identity(observations, "observations")
            self.K = parameters["K"]
            self.D = parameters["D"]
            self.temp = tf.identity(parameters["t_rec"], "temperature")

            self.init_state = tf.nn.log_softmax(parameters["rec_init_state"],
                                                name="initial_state")
            self.trans = parameters["rec_A"]

            self.var = [parameters["rec_init_state"],
                        *parameters["rec_A"].trainable_variables]

    def log_prob_states(self, states):
        NN_input = tf.concat([states[:-1], self.obs[:-1], self.obs[1:]], -1)

        return tf.reduce_sum(tfd.ExpRelaxedOneHotCategorical(
            self.temp, logits=self.trans(NN_input)).log_prob(states[1:]))

    def _update(self, a, x):
        logits = tf.reshape(self.trans(
            tf.reshape(tf.concat([a[0], x[0], x[1]], 0), [1, -1])), [self.K])
        sample = tfd.ExpRelaxedOneHotCategorical(
            self.temp, logits=logits).sample()

        return (sample, logits)

    def sample_states(self, T):
        samples, logits = tf.scan(
            self._update, (self.obs[:-1], self.obs[1:]),
            (self.init_state, tf.zeros([self.K], tf.float32)))

        return tf.concat([tf.reshape(self.init_state, [1, self.K]),
                         samples], 0), logits


def get_dense_NN(name, input_dim, output_dim, layer_sizes, activation="relu",
                 end=False):
    NN = models.Sequential(name)
    NN.add(layers.InputLayer((input_dim,)))
    for num_hidden_units in layer_sizes:
        NN.add(layers.Dense(num_hidden_units, activation))
    if end:
        NN.add(layers.Dense(output_dim, tf.nn.log_softmax))
    else:
        NN.add(layers.Dense(output_dim, "linear"))

    return NN


def get_HHMM_params(num_layers, num_states, obs_dim, gen_layer_sizes,
                    rec_layer_sizes, t_gen, t_rec):
    """Returns a dictionary of all HHMM model parameters
    """

    with tf.variable_scope("model_params"):
        with tf.variable_scope("generative"):
            gen_init_states = []
            gen_As = []
            gen_ends = []
            for l in range(num_layers):
                gen_init_states.append(tf.get_variable(
                    "init_state_{}".format(l + 1), [num_states[l]],
                    tf.float32, tf.random_normal_initializer))

                if l == 0:
                    gen_As.append(get_dense_NN(
                        "trans_NN_{}".format(l + 1),
                        num_states[l] + 1, num_states[0],
                        gen_layer_sizes))
                elif l == num_layers - 1:
                    gen_As.append(get_dense_NN(
                        "trans_NN_{}".format(l + 1),
                        num_states[l - 1] + num_states[l] + 1, num_states[l],
                        gen_layer_sizes))
                else:
                    gen_As.append(get_dense_NN(
                        "trans_NN_{}".format(l + 1),
                        num_states[l - 1] + num_states[l] + 2, num_states[l],
                        gen_layer_sizes))

                if l < num_layers - 1:
                    gen_ends.append(get_dense_NN(
                        "end_NN_{}".format(l + 1),
                        num_states[l] + num_states[l + 1] + 1, 2,
                        gen_layer_sizes, end=True))

            loc = tf.get_variable(
                "loc", [num_states[-1], obs_dim], tf.float32,
                tf.random_normal_initializer)
            scale_tril = tf.get_variable(
                "scale_tril", [num_states[-1], obs_dim * (obs_dim + 1) / 2],
                tf.float32, tf.random_normal_initializer)

        with tf.variable_scope("recognition"):
            rec_init_states = []
            rec_As = []
            rec_ends = []
            for l in range(num_layers):
                rec_init_states.append(tf.get_variable(
                    "init_state_{}".format(l + 1), [num_states[l]],
                    tf.float32, tf.random_normal_initializer))

                if l == 0:
                    rec_As.append(get_dense_NN(
                        "trans_NN_{}".format(l + 1),
                        num_states[0] + 1 + obs_dim * 2,
                        num_states[0], rec_layer_sizes))
                elif l == num_layers - 1:
                    rec_As.append(get_dense_NN(
                        "trans_NN_{}".format(l + 1),
                        num_states[l - 1] + num_states[l] + 1 + obs_dim * 2,
                        num_states[l], rec_layer_sizes))
                else:
                    rec_As.append(get_dense_NN(
                        "trans_NN_{}".format(l + 1),
                        num_states[l - 1] + num_states[l] + 2 + obs_dim * 2,
                        num_states[l], rec_layer_sizes))

                if l < num_layers - 1:
                    rec_ends.append(get_dense_NN(
                        "end_NN_{}".format(l + 1),
                        num_states[l] + num_states[l + 1] + 1 + obs_dim * 2,
                        2, rec_layer_sizes, end=True))

        params = dict(
            L=num_layers, K=num_states, D=obs_dim, t_gen=t_gen, t_rec=t_rec,
            gen_init_states=gen_init_states, gen_As=gen_As, gen_ends=gen_ends,
            loc=loc, scale_tril=scale_tril,
            rec_init_states=rec_init_states, rec_As=rec_As, rec_ends=rec_ends)

    return params
