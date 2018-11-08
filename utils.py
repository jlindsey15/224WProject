import numpy as np


def filter_by(data, key, val):
    mask = np.isin(data[key], val)
    # print(mask.shape)
    for k, v in data.items():
        if k != 'neuron_locations' and k != 'name':
            # print(k, v.shape)
            data[k] = v[mask]
    return data


def cos_diff(v1, v2):
    if np.sum(np.abs(v1)) == 0 or np.sum(np.abs(v2)) == 0:
        return 0
    return np.sum(v2 * v1) / np.sqrt(np.sum(v2 * v2)) / np.sqrt(np.sum(v1 * v1))


def differentiate(arr):
    return arr[:, 1:, :] - arr[:, :-1, :]

def binned(values, value_range):
    counts = np.array([0 for x in value_range])
    for value in values:
        for i in range(len(value_range)):
            if value > value_range[i]:
                counts[i] += 1
    counts = counts[:-1] - counts[1:]
    return counts
