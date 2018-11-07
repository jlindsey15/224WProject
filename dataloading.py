import numpy as np
import pickle
import utils as U
import copy
from benedict import BeneDict


def read_from_files(savenames):
    all_data = []
    for savename in savenames:
        # savename = 'BAYLORGC4_2018_03_23.mat'
        filehandler = open('DataPreprocessed/'+savename+'dataset_L.obj', 'rb')
        dataset_L = pickle.load(filehandler)
        filehandler = open('DataPreprocessed/'+savename+'dataset_R.obj', 'rb')
        dataset_R = pickle.load(filehandler)

        stim_site = dataset_L.pa.train_data.stim_site
        # (Number of time steps, number of trials, number of neurons)

        L_train_rates = dataset_L.pa.train_rates.transpose([1, 0, 2])  # trial x time x neuron
        R_train_rates = dataset_R.pa.train_rates.transpose([1, 0, 2])
        neuron_locations = np.concatenate([np.zeros([L_train_rates.shape[2]]), np.ones([R_train_rates.shape[2]])])
        train_rates = np.concatenate([L_train_rates, R_train_rates], axis=2)
        L_behavior_report = dataset_L.pa.train_data.behavior_report
        R_behavior_report = dataset_R.pa.train_data.behavior_report
        assert(np.sum(L_behavior_report != R_behavior_report) == 0)
        behavior_report = L_behavior_report
        L_behavior_report_type = dataset_L.pa.train_data.behavior_report_type
        R_behavior_report_type = dataset_R.pa.train_data.behavior_report_type
        assert(np.sum(L_behavior_report_type != R_behavior_report_type) == 0)
        behavior_report_type = L_behavior_report_type
        L_task_trial_type = dataset_L.pa.train_data.task_trial_type
        R_task_trial_type = dataset_R.pa.train_data.task_trial_type
        assert(np.sum(L_task_trial_type != R_task_trial_type) == 0)
        task_trial_type = L_task_trial_type

        data = {
            'name': savename,
            'stim_site': stim_site,
            'neuron_locations': neuron_locations,
            'train_rates': train_rates,
            'behavior_report': behavior_report,
            'behavior_report_type': behavior_report_type,
            'task_trial_type': task_trial_type,
        }
        all_data.append(data)
    return all_data


def filter_data(data, perturbation_type=None, enforce_task_success=True, time_window=None):
    data_block = copy.deepcopy(data)
    if perturbation_type is not None:
        data_block = U.filter_by(data_block, 'stim_site', [perturbation_type])
    if enforce_task_success:
        data_block = U.filter_by(data_block, 'behavior_report', [1])
    if time_window is not None:
        low, high = time_window
        data_block['train_rates'] = data_block['train_rates'][:, np.arange(low, high), :]
    data = BeneDict(data_block)
    return data


def normalize_by_behavior_report_type(data):
    data = copy.deepcopy(data)
    # The mice did left/right
    left_mask = data.behavior_report_type == 'l'
    right_mask = data.behavior_report_type == 'r'
    # data.behavior_report

    # Take the mean of left and right
    for mask in [left_mask, right_mask]:
        train_rates_selected = data.train_rates[mask]
        # train_rates_selected = differentiate(train_rates_selected)
        train_rates_selected_mean = np.mean(train_rates_selected, axis=0, keepdims=True)
        data.train_rates[mask] -= train_rates_selected_mean
    return data