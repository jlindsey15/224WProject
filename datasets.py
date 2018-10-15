import torch.utils.data as data
import torch
import torch.nn as nn
import numpy as np
import os
import sys
os.getcwd()
sys.path.append('')

from neupop import Session as Session
from neupop import PopulationAnalysis as PA
from neupop import MultiSessionAnalysis as MS
from neupop import decoders as ld
from neupop import session_parsers as parsers

class DualALMDataset(data.Dataset):
    def __init__(self, train, neuron_split=1.0, back_half_neurons=False, normalize=True, location='left_ALM', train_stim_type='no stim', test_stim_type=None, filename='../../Data/DualALMRecordinbgsDataSet2018_04_14/BAYLORGC12/BAYLORGC12_2018_04_04.mat', only_correct=True):
        # Determine normalizing factor, which is the maximum value across both train and test set.

        sd = Session(filename, parser=parsers.ParserNuoLiDualALM)
        self.train = train
        pa = PA(sd)

        (start, end) = pa.score_input_trial_range()
        pa.cull_data(start, end)
        if train:
            pa.load_train_and_test(train_stim_type=train_stim_type, test_stim_type=None, train_proportion=0.8, location=location)
        else:
            pa.load_train_and_test(train_stim_type=train_stim_type, test_stim_type=None, train_proportion=0.8, location=location)

        if only_correct:
            correct_mask = pa.input_data.behavior_report == 1
            self.fraction_correct = np.sum(pa.train_data.behavior_report) / len(pa.train_data.behavior_report)
            #print(filename, train_stim_type, location, train)
            #print('Frac correct', self.fraction_correct)
            self.behavior_report = pa.train_data.behavior_report
            if train_stim_type != 'early delay bi ALM':
                pa.filter_train_trials_from_input(correct_mask)
                pa.filter_test_trials_from_input(correct_mask)

        pa.preprocess(bin_width=0.4, label_by_report=True, even_test_classes=False)

        train_data = pa.train_rates
        test_data = pa.test_rates
        if train:
            self.norm_factor = np.amax(train_data)
        else:
            self.norm_factor = max(np.amax(train_data), np.amax(test_data))
        self.train_label = pa.train_label
        self.test_label = pa.test_label

        if not normalize:
            self.norm_factor = 1


        self.pa = pa
        if True:#train_stim_type == 'no stim':
            self.coding_directions = [pa.compute_CD_for_each_bin()[0][i].W for i in range(len(pa.compute_CD_for_each_bin()[0]))]

        if train:
            self.data = train_data / self.norm_factor
            self.task_trial_type= pa.train_data.task_trial_type
            self.pole_on_time = np.mean(pa.train_data.task_pole_on_time)
            self.pole_off_time = np.mean(pa.train_data.task_pole_off_time)
            self.cue_on_time = np.mean(pa.train_data.task_cue_on_time)
            self.behavior_report_type = pa.train_data.behavior_report_type

  
 
        else:
            self.data = test_data / self.norm_factor
            self.task_trial_type= pa.test_data.task_trial_type
            self.pole_on_time = np.mean(pa.test_data.task_pole_on_time)
            self.pole_off_time = np.mean(pa.test_data.task_pole_off_time)
            self.cue_on_time = np.mean(pa.test_data.task_cue_on_time)
            self.behavior_report_type = pa.test_data.behavior_report_type

        #self.data = self.data.astype(float)
        #print('data shape', self.data.shape)
        if back_half_neurons:
            self.data = self.data[:, :, (int)(neuron_split*self.data.shape[2]):]
        else:
            self.data = self.data[:, :, :(int)(neuron_split*self.data.shape[2])]
        #print('data shape', self.data.shape)
        self.labels = self.data
        
        # Note that for sequence datasets like this, data has shape (seq_len, n_data, input_size).
        assert self.data.shape[1] == self.labels.shape[1]

    def get_data_max(self):
        return np.amax(self.data)

    def get_data_min(self):
        return np.amin(self.data)

    def get_norm_factor(self):
        return self.norm_factor

    def __getitem__(self, index):
        # Note that for sequence datasets like this, data has shape (seq_len, n_data, input_size).
        x = self.data[:,index,:]
        trial_types = np.array([1 if item == 'l' else -1 for item in self.task_trial_type])
        if self.train:
            if self.pa.train_data.stim_period[index] == 0:
                stim_types =  [[1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.], [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]]
            elif self.pa.train_data.stim_period[index] == 3:
                if self.pa.train_data.stim_site[index] == 1:
                    stim_types = [[1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0.75, 0.5, 0.25, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.25, 0.5, 0.75, 1., 1., 1., 1., 1.], [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]]
                elif self.pa.train_data.stim_site[index] == 2:
                    stim_types = [[1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.], [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0.75, 0.5, 0.25, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.25, 0.5, 0.75, 1., 1., 1., 1., 1.]]
                elif self.pa.train_data.stim_site[index] == 3:
                    stim_types = [[1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0.75, 0.5, 0.25, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.25, 0.5, 0.75, 1., 1., 1., 1., 1.], [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0.75, 0.5, 0.25, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.25, 0.5, 0.75, 1., 1., 1., 1., 1.]]
        else:
            if self.pa.test_data.stim_period[index] == 0:
                stim_types =  [[1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.], [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]]
            elif self.pa.test_data.stim_period[index] == 3:
                if self.pa.test_data.stim_site[index] == 1:
                    stim_types = [[1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0.75, 0.5, 0.25, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.25, 0.5, 0.75, 1., 1., 1., 1., 1.], [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]]
                elif self.pa.test_data.stim_site[index] == 2:
                    stim_types = [[1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.], [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0.75, 0.5, 0.25, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.25, 0.5, 0.75, 1., 1., 1., 1., 1.]]
                elif self.pa.test_data.stim_site[index] == 3:
                    stim_types = [[1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0.75, 0.5, 0.25, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.25, 0.5, 0.75, 1., 1., 1., 1., 1.], [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0.75, 0.5, 0.25, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.25, 0.5, 0.75, 1., 1., 1., 1., 1.]]
        trial_types = trial_types[index]
        y = self.labels[:,index,:]

        return x, trial_types, stim_types


    def __len__(self):
        # Note that for sequence datasets like this, data has shape (seq_len, n_data, input_size).
        return self.data.shape[1]

    def get_len(self):
        # Note that for sequence datasets like this, data has shape (seq_len, n_data, input_size).
        return self.data.shape[1]

class DualALMSimpleDataset(data.Dataset):
    def __init__(self, train, neuron_split=1.0, back_half_neurons=False, normalize=True, location='left_ALM', train_stim_type='no stim', test_stim_type=None, filename='../../Data/DualALMRecordinbgsDataSet2018_04_14/BAYLORGC12/BAYLORGC12_2018_04_04.mat', only_correct=True, timebins='', indexX = 0):
        # Determine normalizing factor, which is the maximum value across both train and test set.

        sd = Session(filename, parser=parsers.ParserNuoLiDualALM)
        self.train = train
        pa = PA(sd)

        (start, end) = pa.score_input_trial_range()
        pa.cull_data(start, end)
        if train:
            pa.load_train_and_test(train_stim_type=train_stim_type, test_stim_type=None, train_proportion=0.8, location=location)
        else:
            pa.load_train_and_test(train_stim_type=train_stim_type, test_stim_type=None, train_proportion=0.8, location=location)

        if only_correct:
            correct_mask = pa.input_data.behavior_report == 1
            self.fraction_correct = np.sum(pa.train_data.behavior_report) / len(pa.train_data.behavior_report)
            #print(filename, train_stim_type, location, train)
            #print('Frac correct', self.fraction_correct)
            self.behavior_report = pa.train_data.behavior_report
            if True:#train_stim_type != 'early delay bi ALM':
                pa.filter_train_trials_from_input(correct_mask)
                pa.filter_test_trials_from_input(correct_mask)

        pa.preprocess(bin_width=0.4, label_by_report=True, even_test_classes=False)

        train_data = pa.train_rates
        test_data = pa.test_rates

        if train:
            self.norm_factor = np.amax(train_data)
        else:
            self.norm_factor = max(np.amax(train_data), np.amax(test_data))
        self.train_label = pa.train_label
        self.test_label = pa.test_label

        if not normalize:
            self.norm_factor = 1


        self.pa = pa
        if True:#train_stim_type == 'no stim':
            self.coding_directions = [pa.compute_CD_for_each_bin()[0][i].W for i in range(len(pa.compute_CD_for_each_bin()[0]))]

        if train:
            self.data = train_data / self.norm_factor
            self.task_trial_type= pa.train_data.task_trial_type
            self.pole_on_time = np.mean(pa.train_data.task_pole_on_time)
            self.pole_off_time = np.mean(pa.train_data.task_pole_off_time)
            self.cue_on_time = np.mean(pa.train_data.task_cue_on_time)
            self.behavior_report_type = pa.train_data.behavior_report_type

  
 
        else:
            self.data = test_data / self.norm_factor
            self.task_trial_type= pa.test_data.task_trial_type
            self.pole_on_time = np.mean(pa.test_data.task_pole_on_time)
            self.pole_off_time = np.mean(pa.test_data.task_pole_off_time)
            self.cue_on_time = np.mean(pa.test_data.task_cue_on_time)
            self.behavior_report_type = pa.test_data.behavior_report_type

        #self.data = self.data.astype(float)
        #print('data shape', self.data.shape)
        if back_half_neurons:
            self.data = self.data[:, :, (int)(neuron_split*self.data.shape[2]):]
        else:
            self.data = self.data[:, :, :(int)(neuron_split*self.data.shape[2])]
        if timebins == '':
            times = [28, 29, 30, 31, 32, 33, 34, 35]
        elif timebins == 'firstbins':
            times = [28, 29, 30, 31]
        elif timebins == 'lastbins':
            times = [32, 33, 34, 35]
        labeltimes = [time + 1 for time in times]
        num_neurons = self.data.shape[2]
        datatemp = self.data
        self.data = datatemp[times, :, :].reshape(-1, num_neurons) #data is full activityat time t
        #print('data shape', self.data.shape)
        self.labels = datatemp[labeltimes, :, :].reshape(-1, num_neurons) #temp, to be changed later in the code

        self.cd_L = np.expand_dims(self.coding_directions[36][pa.train_data.neural_unit_location=='left_ALM'], 1)
        self.cd_R = np.expand_dims(self.coding_directions[36][pa.train_data.neural_unit_location=='right_ALM'], 1)
        self.cd_L /= np.linalg.norm(self.cd_L)
        self.cd_R /= np.linalg.norm(self.cd_R)
        self.fulllabels = self.labels #fulllabels are full activity at time t+1
        labelsL = np.matmul(self.labels[:, pa.train_data.neural_unit_location=='left_ALM'], self.cd_L)
        labelsR = np.matmul(self.labels[:, pa.train_data.neural_unit_location=='right_ALM'], self.cd_R)
        dataL = np.matmul(self.data[:, pa.train_data.neural_unit_location=='left_ALM'], self.cd_L)
        dataR = np.matmul(self.data[:, pa.train_data.neural_unit_location=='right_ALM'], self.cd_R)
        self.CDdata = np.concatenate([dataL, dataR], 1) #CDdata is CD activity at time t
        self.labels = np.concatenate([labelsL, labelsR], 1) # labels are CD activity at time t+1

        linreg_file = 'bridgeFullToCDPred'
        
        if train:
            self.linreg_L = np.squeeze(np.load(linreg_file+'TRAINL'+'.npy')[indexX])#[:, :, :]#[:, 13:, :]
            self.linreg_R = np.squeeze(np.load(linreg_file+'TRAINR'+'.npy')[indexX])#[:, :, :]#[:, 13:, :]
        else:
            self.linreg_L = np.squeeze(np.load(linreg_file+'TESTL'+'.npy')[indexX])#[:, :, :]#[:, 13:, :]
            self.linreg_R = np.squeeze(np.load(linreg_file+'TESTR'+'.npy')[indexX])#[:, :, :]#[:, 13:, :]

        linreg_file = 'nobridgeFullToCDPred'
        
        if train:
            self.nobridge_linreg_L = np.squeeze(np.load(linreg_file+'TRAINL'+'.npy')[indexX])#[:, :, :]#[:, 13:, :]
            self.nobridge_linreg_R = np.squeeze(np.load(linreg_file+'TRAINR'+'.npy')[indexX])#[:, :, :]#[:, 13:, :]
        else:
            self.nobridge_linreg_L = np.squeeze(np.load(linreg_file+'TESTL'+'.npy')[indexX])#[:, :, :]#[:, 13:, :]
            self.nobridge_linreg_R = np.squeeze(np.load(linreg_file+'TESTR'+'.npy')[indexX])#[:, :, :]#[:, 13:, :]
        

        #assert self.data.shape[1] == self.labels.shape[1]

    def get_data_max(self):
        return np.amax(self.data)

    def get_data_min(self):
        return np.amin(self.data)

    def get_norm_factor(self):
        return self.norm_factor

    def __getitem__(self, index):
        # Note that for sequence datasets like this, data has shape (seq_len, n_data, input_size).
        #print('My index is ', index)
        x = self.data[index,:]
        y = self.labels[index, :]
        linreg_L = self.linreg_L[index]
        linreg_R = self.linreg_R[index]
        nobridge_linreg_L = self.nobridge_linreg_L[index]
        nobridge_linreg_R = self.nobridge_linreg_R[index]
        return x, y, linreg_L, linreg_R, nobridge_linreg_L, nobridge_linreg_R


    def __len__(self):
        # Note that for sequence datasets like this, data has shape (seq_len, n_data, input_size).
        return self.data.shape[0]

    def get_len(self):
        # Note that for sequence datasets like this, data has shape (seq_len, n_data, input_size).
        return self.data.shape[0]



class DualALMFineDataset(data.Dataset):
    def __init__(self, train, neuron_split=1.0, back_half_neurons=False, normalize=True, location='left_ALM', train_stim_type='no stim', test_stim_type=None, filename='../../Data/DualALMRecordinbgsDataSet2018_04_14/BAYLORGC12/BAYLORGC12_2018_04_04.mat', only_correct=True):
        # Determine normalizing factor, which is the maximum value across both train and test set.

        sd = Session(filename, parser=parsers.ParserNuoLiDualALM)
        self.train = train
        pa = PA(sd)

        (start, end) = pa.score_input_trial_range()
        pa.cull_data(start, end)
        if train:
            pa.load_train_and_test(train_stim_type=train_stim_type, test_stim_type=None, train_proportion=0.8, location=location)
        else:
            pa.load_train_and_test(train_stim_type=train_stim_type, test_stim_type=None, train_proportion=0.8, location=location)

        if only_correct:
            correct_mask = pa.input_data.behavior_report == 1
            self.fraction_correct = np.sum(pa.train_data.behavior_report) / len(pa.train_data.behavior_report)
            #print(filename, train_stim_type, location, train)
            #print('Frac correct', self.fraction_correct)
            self.behavior_report = pa.train_data.behavior_report
            if train_stim_type != 'early delay bi ALM':
                pa.filter_train_trials_from_input(correct_mask)
                pa.filter_test_trials_from_input(correct_mask)

        pa.preprocess(bin_width=0.2, stride=0.05, label_by_report=True, even_test_classes=False)

        train_data = pa.train_rates
        test_data = pa.test_rates
        if train:
            self.norm_factor = np.amax(train_data)
        else:
            self.norm_factor = max(np.amax(train_data), np.amax(test_data))
        self.train_label = pa.train_label
        self.test_label = pa.test_label

        if not normalize:
            self.norm_factor = 1


        self.pa = pa
        if True:#train_stim_type == 'no stim':
            self.coding_directions = [pa.compute_CD_for_each_bin()[0][i].W for i in range(len(pa.compute_CD_for_each_bin()[0]))]

        if train:
            self.data = train_data / self.norm_factor
            self.task_trial_type= pa.train_data.task_trial_type
            self.pole_on_time = np.mean(pa.train_data.task_pole_on_time)
            self.pole_off_time = np.mean(pa.train_data.task_pole_off_time)
            self.cue_on_time = np.mean(pa.train_data.task_cue_on_time)
            self.behavior_report_type = pa.train_data.behavior_report_type

  
 
        else:
            self.data = test_data / self.norm_factor
            self.task_trial_type= pa.test_data.task_trial_type
            self.pole_on_time = np.mean(pa.test_data.task_pole_on_time)
            self.pole_off_time = np.mean(pa.test_data.task_pole_off_time)
            self.cue_on_time = np.mean(pa.test_data.task_cue_on_time)
            self.behavior_report_type = pa.test_data.behavior_report_type

        #self.data = self.data.astype(float)
        #print('data shape', self.data.shape)
        if back_half_neurons:
            self.data = self.data[:, :, (int)(neuron_split*self.data.shape[2]):]
        else:
            self.data = self.data[:, :, :(int)(neuron_split*self.data.shape[2])]
        #print('data shape', self.data.shape)
        self.labels = self.data
        
        # Note that for sequence datasets like this, data has shape (seq_len, n_data, input_size).
        assert self.data.shape[1] == self.labels.shape[1]

    def get_data_max(self):
        return np.amax(self.data)

    def get_data_min(self):
        return np.amin(self.data)

    def get_norm_factor(self):
        return self.norm_factor

    def __getitem__(self, index):
        # Note that for sequence datasets like this, data has shape (seq_len, n_data, input_size).
        x = self.data[:,index,:]
        trial_types = np.array([1 if item == 'l' else -1 for item in self.task_trial_type])
        if self.train:
            if self.pa.train_data.stim_period[index] == 0:
                stim_types =  [[1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.], [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]]
            elif self.pa.train_data.stim_period[index] == 3:
                if self.pa.train_data.stim_site[index] == 1:
                    stim_types = [[1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0.75, 0.5, 0.25, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.25, 0.5, 0.75, 1., 1., 1., 1., 1.], [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]]
                elif self.pa.train_data.stim_site[index] == 2:
                    stim_types = [[1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.], [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0.75, 0.5, 0.25, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.25, 0.5, 0.75, 1., 1., 1., 1., 1.]]
                elif self.pa.train_data.stim_site[index] == 3:
                    stim_types = [[1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0.75, 0.5, 0.25, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.25, 0.5, 0.75, 1., 1., 1., 1., 1.], [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0.75, 0.5, 0.25, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.25, 0.5, 0.75, 1., 1., 1., 1., 1.]]
        else:
            if self.pa.test_data.stim_period[index] == 0:
                stim_types =  [[1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.], [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]]
            elif self.pa.test_data.stim_period[index] == 3:
                if self.pa.test_data.stim_site[index] == 1:
                    stim_types = [[1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0.75, 0.5, 0.25, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.25, 0.5, 0.75, 1., 1., 1., 1., 1.], [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]]
                elif self.pa.test_data.stim_site[index] == 2:
                    stim_types = [[1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.], [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0.75, 0.5, 0.25, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.25, 0.5, 0.75, 1., 1., 1., 1., 1.]]
                elif self.pa.test_data.stim_site[index] == 3:
                    stim_types = [[1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0.75, 0.5, 0.25, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.25, 0.5, 0.75, 1., 1., 1., 1., 1.], [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0.75, 0.5, 0.25, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.25, 0.5, 0.75, 1., 1., 1., 1., 1.]]
        trial_types = trial_types[index]
        y = self.labels[:,index,:]

        return x, trial_types, stim_types


    def __len__(self):
        # Note that for sequence datasets like this, data has shape (seq_len, n_data, input_size).
        return self.data.shape[1]

    def get_len(self):
        # Note that for sequence datasets like this, data has shape (seq_len, n_data, input_size).
        return self.data.shape[1]




