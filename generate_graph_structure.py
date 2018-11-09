import numpy as np
from scipy.stats import linregress
from scipy.stats.stats import pearsonr
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from dataloading import normalize_by_behavior_report_type



def create_adjacency_matrix(data, edge_weight_func, perturbation_type):
    if perturbation_type != 3 and (edge_weight_func == flatten_correlation or edge_weight_func == granger_causality):
        print('normalizing')
        data = normalize_by_behavior_report_type(data)
    train_rates = data.train_rates
    N, T, K = train_rates.shape
    print('N, T, K: ', N, T, K)
    print(data.neuron_locations.shape)
    mat = np.zeros([K, K])
    for k1 in range(K):
        for k2 in range(K):
            if k1 == k2:
                continue
            v1 = train_rates[:, :, k1]
            v2 = train_rates[:, :, k2]
            value = edge_weight_func(v1, v2)
            mat[k1, k2] = value
    return mat


def flatten_correlation(v1, v2):
    # v1 is of shape trial x time_step
    v1 = v1.flatten()
    v2 = v2.flatten()
    value, _ = np.abs(pearsonr(v1, v2))
    return value


def granger_causality(v1, v2):
    """
        The probability that v2 => v1 is a causal relation
    """
    signal1 = v1[:, :-1].flatten()
    signal2 = v2[:, :-1].flatten()
    response = np.array(v2[:, 1:]).flatten()
    # note signal 1 is used to predict the residue
    # because we want to know whether v1 causes v2
    slope1, intercept1, r_value1, p_value1, std_err1 \
        = linregress(signal2, response)
    residue = response - (signal2.flatten() * slope1 + intercept1)
    slope2, intercept2, r_value2, p_value2, std_err2 \
        = linregress(signal1, residue)
    return 1 - p_value2

def behavioral_prediction_correlation_wrapper(behavior_report_type):
    def behavioral_prediction_correlation(v1, v2, behavior_report_type=behavior_report_type):
        behavior_report_type = behavior_report_type == 'l'
        v1last = v1[:, -1].reshape(-1, 1)
        v2last = v2[:, -1].reshape(-1, 1)
        v1reg = LogisticRegression()
        v1reg.fit(v1last, behavior_report_type)
        v1pred = v1reg.predict(v1last)
        v2reg = LogisticRegression()
        v2reg.fit(v2last, behavior_report_type)
        v2pred = v2reg.predict(v2last)
        return(np.sum(v1pred == v2pred) / len(v1pred))
    return behavioral_prediction_correlation
