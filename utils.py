import numpy as np


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
