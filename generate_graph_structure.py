import numpy as np
from scipy.stats import linregress
from scipy.stats.stats import pearsonr


def create_adjacency_matrix(data, edge_weight_func):
    train_rates = data.train_rates
    N, T, K = train_rates.shape
    print('N, T, K: ', N, T, K)
    print(data.neuron_locations.shape)
    mat = np.zeros([K, K])
    for k1 in range(K):
        for k2 in range(K):
            if k1 == k2:
                continue
            v1 = train_rates[:, :, k1].flatten()
            v2 = train_rates[:, :, k2].flatten()
            value = edge_weight_func(v1, v2)
            mat[k1, k2] = value
    return mat


def flatten_correlation(v1, v2):
    # v1 is of shape trial x time_step
    v1 = v1.flatten()
    v2 = v2.flatten()
    value, _ = pearsonr(v1, v2)
    return value


def granger_causality(v1, v2, degree=1):
    """
        The probability that v2 => v1 is a causal relation
    """
    signal1 = []
    signal2 = []
    for i in range(degree):
        signal1.append(v1[i:-(degree - i)])
        signal2.append(v2[i:-(degree - i)])
    signal1 = np.stack(signal1, axis=1)
    signal2 = np.stack(signal2, axis=1)
    response = np.array(v2[degree:])
    # note signal 1 is used to predict the residue
    # because we want to know whether v1 causes v2
    slope1, intercept1, r_value1, p_value1, std_err1 \
        = linregress(signal2.flatten(), response) 
    residue = response - (signal2.flatten() * slope1 + intercept1)
    slope2, intercept2, r_value2, p_value2, std_err2 \
        = linregress(signal1.flatten(), residue)
    return 1 - p_value2
