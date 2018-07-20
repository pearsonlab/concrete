import tensorflow as tf
import numpy as np
from edward.models import (MultivariateNormalTriL, RandomVariable,
                           RelaxedOneHotCategorical)
from tensorflow.contrib.distributions import (Distribution, fill_triangular,
                                              FULLY_REPARAMETERIZED)
from tensorflow.contrib.keras import layers, models


def sample_HMM(parameters, n, seed=None):
    """Samples from an HMM (Gaussian outputs) with provided parameters.

    Args:
        parameters: A dictionary of HMM parameters.
            Keys include: n_states (number of states),
            init_prob (initial distribution of states),
            trans_matrix (matrix of state transition probabilities, each row
                sums to 1),
            obs_dim (dimension of observations),
            mean (mean of the emission distribution for every state),
            cov (covariance matrix of the emission distribution).
        n: Number of observations to sample.
    """

    K = parameters["n_states"]
    pi_0 = parameters["init_prob"]
    A = parameters["trans_matrix"]

    D = parameters["obs_dim"]
    mean = parameters["mean"]
    cov = parameters["cov"]

    np.random.seed(seed)

    # create empty numpy arrays to store samples
    states = np.empty(n, np.int32)
    obs = np.empty((n, D), np.float32)

    for t in range(n):
        if t == 0:
            # sample the first state from initial distribution
            states[t] = np.random.choice(K, p=pi_0)
        else:
            # get the next state based on transition matrix (the row
            # corresponding to the previous state)
            states[t] = np.random.choice(K, p=A[states[t - 1]])

        # sample observation from the corresponding Gaussian distribution
        obs[t] = np.random.multivariate_normal(
            mean[states[t]], cov[states[t]])

    return states, obs


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
        NN = models.Sequential(name="recognition_neural_network")
        NN.add(layers.Dense(layer_dim, input_dim=obs_dim * 3,
                            activation="relu", name="layer_1"))
        if n_layers > 2:
            for i in range(2, n_layers):
                NN.add(layers.Dense(layer_dim, activation="relu",
                                    name="layer_{}".format(i)))
        NN.add(layers.Dense(n_states, activation="softmax",
                            name="layer_{}".format(n_layers)))

        params = dict(
            K=n_states, D=obs_dim,
            t_gen=temp_gen, t_rec=temp_rec,
            unc_pi_0=tf.get_variable(
                "unconstrained_init_prob", [n_states], tf.float32,
                tf.random_normal_initializer),
            unc_A=tf.get_variable(
                "unconstrained_trans_mat", [n_states, n_states], tf.float32,
                tf.random_normal_initializer),
            loc=tf.get_variable(
                "loc", [n_states, obs_dim], tf.float32,
                tf.random_normal_initializer),
            unc_scale_tril=tf.get_variable(
                "unconstrained_scale_tril",
                [n_states, obs_dim * (obs_dim + 1) / 2], tf.float32,
                tf.random_normal_initializer),
            NN=NN)

    return params


class HMM_gen(RandomVariable, Distribution):
    """Random variable class for HMM (generative model)
    """

    def __init__(self, observations, parameters, *args, **kwargs):
        name = kwargs.get("name", "generative")
        with tf.name_scope(name):
            self.obs = tf.identity(observations, "observations")
            self.K = parameters["K"]
            self.D = parameters["D"]
            self.temp = tf.identity(parameters["t_gen"], "temperature")

            # normalize the probability of initial distribution
            self.pi_0 = tf.nn.softmax(
                parameters["unc_pi_0"], name="initial_probability")
            self.init_dist = RelaxedOneHotCategorical(
                self.temp, probs=self.pi_0, name="initial_distribution")

            # normalize the probability of transition distributions
            self.A = tf.nn.softmax(
                parameters["unc_A"], name="transition_matrix")
            self.trans = RelaxedOneHotCategorical(
                self.temp, probs=self.A, name="transition")

            self.loc = tf.identity(parameters["loc"], name="loc")
            # cholesky factor of covariance matrix
            self.scale_tril = fill_triangular(tf.nn.softplus(
                parameters["unc_scale_tril"]), name="scale_tril")
            self.cov = tf.matmul(
                self.scale_tril, tf.matrix_transpose(self.scale_tril),
                name="cov")
            self.em = MultivariateNormalTriL(
                self.loc, self.scale_tril, name="emission")

            self.var = [parameters["unc_pi_0"], parameters["unc_A"],
                        parameters["loc"], parameters["unc_scale_tril"]]

        if "name" not in kwargs:
            kwargs["name"] = name
        if "dtype" not in kwargs:
            kwargs["dtype"] = tf.float32
        if "reparameterization_type" not in kwargs:
            kwargs["reparameterization_type"] = FULLY_REPARAMETERIZED
        if "validate_args" not in kwargs:
            kwargs["validate_args"] = True
        if "allow_nan_stats" not in kwargs:
            kwargs["allow_nan_stats"] = False

        super(HMM_gen, self).__init__(*args, **kwargs)

        self._args = (observations, parameters)

    def _log_prob(self, value):
        log_prob_0 = tf.add(
            tf.reduce_logsumexp(tf.add(
                self.em.log_prob(self.obs[0]), tf.log(value[0]))),
            self.init_dist.log_prob(value[0]))

        log_prob = tf.scan(
            lambda a, x: tf.add_n(
                [tf.reduce_logsumexp(tf.add(
                    self.em.log_prob(x[0]), tf.log(x[1]))),
                 tf.reduce_logsumexp(tf.add(
                    self.trans.log_prob(x[1]), tf.log(x[2]))), a]),
            (self.obs[1:], value[1:], value[:-1]), log_prob_0)

        return log_prob


class HMM_rec(RandomVariable, Distribution):
    """Random variable class for HMM (recognition model)
    """

    def __init__(self, observations, parameters, *args, **kwargs):
        name = kwargs.get("name", "recognition")
        with tf.name_scope(name):
            self.obs = tf.identity(observations, "observations")
            self.K = parameters["K"]
            self.D = parameters["D"]
            self.temp = tf.identity(parameters["t_rec"], "temperature")

            self.NN = parameters["NN"]
            # inputs to the neural network include the previous, current,
            # and next observations
            self.NN_inputs = tf.concat(
                [tf.concat(
                    [tf.expand_dims(self.obs[0], 0, "first"), self.obs[:-1]],
                    0, "previous"),
                 self.obs,
                 tf.concat(
                    [self.obs[1:], tf.expand_dims(self.obs[-1], 0, "last")],
                    0, "next")],
                -1, "NN_inputs")
            self.probs = tf.identity(
                self.NN(self.NN_inputs), "state_probability")
            self.dist = RelaxedOneHotCategorical(
                self.temp, probs=self.probs, name="state_distribution")

            self.var = self.NN.variables

        if "name" not in kwargs:
            kwargs["name"] = name
        if "dtype" not in kwargs:
            kwargs["dtype"] = tf.float32
        if "reparameterization_type" not in kwargs:
            kwargs["reparameterization_type"] = FULLY_REPARAMETERIZED
        if "validate_args" not in kwargs:
            kwargs["validate_args"] = True
        if "allow_nan_stats" not in kwargs:
            kwargs["allow_nan_stats"] = False

        super(HMM_rec, self).__init__(*args, **kwargs)

        self._args = (observations, parameters)

    def _log_prob(self, value):
        return tf.reduce_sum(self.dist.log_prob(value))

    def _sample_n(self, n, seed=None):
        return self.dist.sample(n, seed)
