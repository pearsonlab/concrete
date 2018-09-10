import numpy as np


def sample_HMM(parameters, T, seed=None):
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
        T: Number of observations to sample.
    """

    K = parameters["num_states"]
    pi_0 = parameters["init_prob"]
    A = parameters["trans_matrix"]

    D = parameters["obs_dim"]
    mean = parameters["mean"]
    cov = parameters["cov"]

    np.random.seed(seed)

    # create empty numpy arrays to store samples
    states = np.empty(T, np.int32)
    obs = np.empty((T, D), np.float32)

    for t in range(T):
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


def sample_HHMM(parameters, T, seed=None):
    L = parameters["n_layers"]
    K = parameters["n_states"]
    pi_0 = parameters["init_prob"]

    # the last term of each row is the probability of reaching end state
    # (except the top layer)
    A = parameters["trans_matrix"]

    D = parameters["obs_dim"]
    mean = parameters["mean"]
    cov = parameters["cov"]

    np.random.seed(seed)

    # indicator for whether the current state at each layer ends
    # (whether the child layer reaches end state)
    end = np.full(L, False)
    states = np.empty((T, L), np.int32)
    obs = np.empty((T, D), np.float32)

    for t in range(T):
        if t == 0:
            # at the first time point, draw from the initial distribution
            # conditioned on the parent layer (except the top layer)
            for l in range(L):
                if l == 0:
                    states[t, l] = np.random.choice(K[l], p=pi_0[l][0])
                else:
                    states[t, l] = np.random.choice(
                        K[l], p=pi_0[l][states[t, l - 1]])

            # the last layer (with production states) always transitions
            end[-1] = True

        else:
            for l in range(L):
                # the top layer either transitions or not
                if l == 0:
                    if not end[l]:
                        states[t, l] = states[t - 1, l]
                    else:
                        states[t, l] = np.random.choice(
                            K[l], p=A[l][0][states[t - 1, l]])

                else:
                    # no transition if the child layer doesn't reach end state
                    if not end[l]:
                        states[t, l] = states[t - 1, l]

                    # transition conditioned on the parent layer if end state
                    # not reached at t - 1
                    elif not end[l - 1]:
                        states[t, l] = np.random.choice(
                            K[l],
                            p=A[l][states[t, l - 1]][states[t - 1, l]][:-1])

                    # resample from the initial distribution conditioned on
                    # the parent layer if end state reached at t - 1
                    else:
                        states[t, l] = np.random.choice(
                            K[l], p=pi_0[l][states[t, l - 1]])

            for l in range(L - 1, 0, -1):
                # reach end state with some probability conditioned on
                # the parent layer only if the child layer reaches end state
                if (end[l] and np.random.uniform()
                    <= A[l][states[t, l - 1]][states[t, l]][-1]):
                    end[l - 1] = True

        # sample observation from the corresponding Gaussian distribution
        obs[t] = np.random.multivariate_normal(
            mean[states[t, -1]], cov[states[t, -1]])

    return states, obs