from __future__ import print_function
import argparse
import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
import numpy as np
import os
import shutil
import time
import math
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import pickle


os.environ["CUDA_VISIBLE_DEVICES"] = "0,"


from datasets import DualALMDataset


# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--batch-size', type=int, default=2, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 100)')
parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                    help='learning rate')
parser.add_argument('--weight-decay', type=float, default=0.0, help='L2 regularization coefficient.')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=40, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--model', default='lfads', help='lfads')
parser.add_argument('--model-size', default='size1', help='size[#]')
parser.add_argument('--location', default=None, help='None, left_ALM, right_ALM')
parser.add_argument('--stim_type', default='no stim', help='no stim, early delay left ALM, early delay right ALM, early delay bi ALM')
parser.add_argument('--filename', default=None)
parser.add_argument('--eval_on', default='postperturbation')
parser.add_argument('--savename', default=None)
parser.add_argument('--test-trained-model', default='', type=str, help='Path to the saved trained model.')
parser.add_argument('--resume', default='', type=str, help='path to latest checkpoint (default: none)')
parser.add_argument('--bridge', default="", type=str)
parser.add_argument('--source', default=None)
parser.add_argument('--target', default=None)
parser.add_argument('--save-model', action='store_true', default=True)
parser.add_argument('--hp-search', action='store_true', default=False)
parser.add_argument('--gpu-idx', type=int, default=0)
parser.add_argument('--f2r_nonlins', type=int, default=0)
parser.add_argument('--non_variational', action='store_true', default=False)
parser.add_argument('--provide_trial_type', action='store_true', default=False)
parser.add_argument('--no_ext_inp', action='store_true', default=False)
parser.add_argument('--linear', action='store_true', default=False)
parser.add_argument('--superlinear', action='store_true', default=False)
parser.add_argument('--duperlinear', action='store_true', default=False)
parser.add_argument('--no_f', default='', type=str)
parser.add_argument('--test_no_f', default='', type=str)
parser.add_argument('--all_trials', default='', type=str)
parser.add_argument('--uni_trials', default='', type=str)
parser.add_argument('--nocdsource', default='', type=str)
parser.add_argument('--use_exp', action='store_true', default=False)
parser.add_argument('--no_cuda', action='store_true', default=False)
parser.add_argument('--save_preds', default='', type=str, help='Destination path for trained model preds')
parser.add_argument('--neuron_split', type=float, default=1.0, help='Fraction of neurons to train on')
parser.add_argument('--f', type=int, default=10, help='Number of latent factors')

args = parser.parse_args()

if args.bridge == 'full' or args.bridge =='':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,"
if args.bridge == 'all_all' or args.bridge == 'all_CD':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1,"
if args.bridge == 'to_CD' or args.bridge == 'from_CD':
    os.environ["CUDA_VISIBLE_DEVICES"] = "2,"


args.cuda = not args.no_cuda and torch.cuda.is_available()

filenames = ['Data/DualALMRecordinbgsDataSet2018_04_14/BAYLORGC4/BAYLORGC4_2018_03_22.mat', 'Data/DualALMRecordinbgsDataSet2018_04_14/BAYLORGC4/BAYLORGC4_2018_03_21.mat', 'Data/DualALMRecordinbgsDataSet2018_04_14/BAYLORGC4/BAYLORGC4_2018_03_23.mat', 'Data/DualALMRecordinbgsDataSet2018_04_14/BAYLORGC4/BAYLORGC4_2018_03_20.mat', 'Data/DualALMRecordinbgsDataSet2018_04_14/BAYLORGC4/BAYLORGC4_2018_03_25.mat', 'Data/DualALMRecordinbgsDataSet2018_04_14/BAYLORGC12/BAYLORGC12_2018_04_03.mat', 'Data/DualALMRecordinbgsDataSet2018_04_14/BAYLORGC12/BAYLORGC12_2018_04_04.mat', 'Data/DualALMRecordinbgsDataSet2018_04_14/BAYLORGC12/BAYLORGC12_2018_04_05.mat', 'Data/DualALMRecordinbgsDataSet2018_04_14/BAYLORGC12/BAYLORGC12_2018_04_06.mat', 'Data/DualALMRecordinbgsDataSet2018_04_14/BAYLORGC13/BAYLORGC13_2018_04_23.mat', 'Data/DualALMRecordinbgsDataSet2018_04_14/BAYLORGC13/BAYLORGC13_2018_04_24.mat', 'Data/DualALMRecordinbgsDataSet2018_04_14/BAYLORGC13/BAYLORGC13_2018_04_25.mat', 'Data/DualALMRecordinbgsDataSet2018_04_14/BAYLORGC15/BAYLORGC15_2018_05_07.mat', 'Data/DualALMRecordinbgsDataSet2018_04_14/BAYLORGC15/BAYLORGC15_2018_05_08.mat', 'Data/DualALMRecordinbgsDataSet2018_04_14/BAYLORGC15/BAYLORGC15_2018_05_09.mat', 'Data/DualALMRecordinbgsDataSet2018_04_14/BAYLORGC15/BAYLORGC15_2018_05_10.mat', 'Data/DualALMRecordinbgsDataSet2018_04_14/BAYLORGC15/BAYLORGC15_2018_05_11.mat', 'Data/DualALMRecordinbgsDataSet2018_04_14/BAYLORGC17/BAYLORGC17_2018_06_08.mat', 'Data/DualALMRecordinbgsDataSet2018_04_14/BAYLORGC17/BAYLORGC17_2018_06_11.mat', 'Data/DualALMRecordinbgsDataSet2018_04_14/BAYLORGC18/BAYLORGC18_2018_05_31.mat', 'Data/DualALMRecordinbgsDataSet2018_04_14/BAYLORGC19/BAYLORGC19_2018_06_20.mat', 'Data/DualALMRecordinbgsDataSet2018_04_14/BAYLORGC19/BAYLORGC19_2018_06_21.mat', 'Data/DualALMRecordinbgsDataSet2018_04_14/BAYLORGC19/BAYLORGC19_2018_06_22.mat']

savenames = ['BAYLORGC4_2018_03_22.mat', 'BAYLORGC4_2018_03_21.mat', 'BAYLORGC4_2018_03_23.mat', 'BAYLORGC4_2018_03_20.mat', 'BAYLORGC4_2018_03_25.mat', 'BAYLORGC12_2018_04_03.mat', 'BAYLORGC12_2018_04_04.mat', 'BAYLORGC12_2018_04_05.mat', 'BAYLORGC12_2018_04_06.mat', 'BAYLORGC13_2018_04_23.mat', 'BAYLORGC13_2018_04_24.mat', 'BAYLORGC13_2018_04_25.mat', 'BAYLORGC15_2018_05_07.mat', 'BAYLORGC15_2018_05_08.mat', 'BAYLORGC15_2018_05_09.mat', 'BAYLORGC15_2018_05_10.mat', 'BAYLORGC15_2018_05_11.mat', 'BAYLORGC17_2018_06_08.mat', 'BAYLORGC17_2018_06_11.mat', 'BAYLORGC18_2018_05_31.mat', 'BAYLORGC19_2018_06_20.mat', 'BAYLORGC19_2018_06_21.mat', 'BAYLORGC19_2018_06_22.mat']

epoch_dict = {'presample': range(0, 9), 'sample': range(8, 21), 'presamplesample': range(0, 21), 'delay': range(20, 37), 'sampledelay': range(8, 37), 'task': range(8, 55), 'perturbation': range(24, 28), 'preduringperturbation': range(0, 26), 'postperturbation': range(27, 37), 'presamplesampledelay': range(0, 37)}


def main(args):



    #dataset_L = DualALMDataset(train=True, neuron_split=args.neuron_split, location='left_ALM', filename=args.filename, train_stim_type=None, only_correct=False)
    #dataset_R = DualALMDataset(train=True, neuron_split=args.neuron_split, location='right_ALM', filename=args.filename, train_stim_type=None, only_correct=False) 
    #filehandler = open('DataPreprocessed/'+args.savename+'dataset_L.obj', 'wb')
    #pickle.dump(dataset_L, filehandler)
    #filehandler = open('DataPreprocessed/'+args.savename+'dataset_R.obj', 'wb')
    #pickle.dump(dataset_R, filehandler)
    #Comment out the above lines once you've run this once
    filehandler = open('DataPreprocessed/'+args.savename+'dataset_L.obj', 'rb')
    dataset_L = pickle.load(filehandler) 
    filehandler = open('DataPreprocessed/'+args.savename+'dataset_R.obj', 'rb')
    dataset_R = pickle.load(filehandler) 

    mask = dataset_L.pa.train_data.stim_site == 0
    dataset_L.pa.train_rates = dataset_L.pa.train_rates[:, mask]
    dataset_R.pa.train_rates = dataset_R.pa.train_rates[:, mask]
    dataset_L.pa.train_data.behavior_report = dataset_L.pa.train_data.behavior_report[mask]
    dataset_R.pa.train_data.behavior_report = dataset_R.pa.train_data.behavior_report[mask]
    dataset_L.pa.train_data.behavior_report_type = dataset_L.pa.train_data.behavior_report_type[mask]
    dataset_R.pa.train_data.behavior_report_type = dataset_R.pa.train_data.behavior_report_type[mask]

    print(dataset_L.pa.train_data.behavior_report.shape)
    classifier_L = dataset_L.pa.classifiers[0][36]
    classifier_R = dataset_R.pa.classifiers[0][36]
    preds_L = classifier_L.classify(dataset_L.pa.train_rates[36])
    preds_R = classifier_R.classify(dataset_R.pa.train_rates[36])
    proj_L = classifier_L.project_to_W(dataset_L.pa.train_rates[36])
    proj_R = classifier_R.project_to_W(dataset_R.pa.train_rates[36])
    agree = preds_L == preds_R
    disagree = preds_L != preds_R
    correct = dataset_L.pa.train_data.behavior_report == 1  
    error = dataset_L.pa.train_data.behavior_report == 0
    left = dataset_L.pa.train_data.behavior_report_type == 'l'
    right = dataset_L.pa.train_data.behavior_report_type == 'r'

    print(proj_L[error].shape, proj_L[correct].shape)

    ax = plt.subplot(gs[(indexX+1) // 6, (indexX+1) % 6])

    plt.scatter(proj_L[correct * left], proj_R[correct * left], c='r', marker='.')
    plt.scatter(proj_L[correct * right], proj_R[correct * right], c='b', marker='.')
    plt.scatter(proj_L[error * left], proj_R[error * left], c='r', marker='x')
    plt.scatter(proj_L[error * right], proj_R[error * right], c='b', marker='x')
    plt.title('Session ' + str(indexX+1))

    valL = np.max(np.abs(proj_L))
    valR = np.max(np.abs(proj_R))
    plt.xlim([-valL, valL])
    plt.ylim([-valR, valR])
    plt.plot([-valL, valL], [0, 0], 'k-')
    plt.plot([0, 0], [-valR, valR], 'k-')






if __name__ == '__main__':
    if args.hp_search:
        hyperparameter_search(args)
    else:
        if args.filename is None:
            fig = plt.figure(figsize=(24, 16))
            gs = gridspec.GridSpec(4,6, wspace=0.3, hspace=0.3)
            plt.gcf().text(0.5, 0.95, 'Control Trials', fontsize=16, rotation='horizontal', horizontalalignment='center')
            ax = plt.subplot(gs[0, 0])
            plt.scatter([], [], c='r', marker='.')
            plt.scatter([], [], c='b', marker='.')
            plt.scatter([], [], c='r', marker='x')
            plt.scatter([], [], c='b', marker='x')
            plt.xticks([])
            plt.yticks([])
            plt.legend(['Correct left', 'Correct right', 'Error left', 'Error right'])
            ax.set_xlabel('Left ALM Decoder Projection', fontsize=8)
            ax.set_ylabel('Right ALM Decoder Projection', fontsize=8)
            for indexX in range(len(filenames)):
                args.filename = filenames[indexX]
                args.savename = savenames[indexX]
                #args.resume = 'Default'
                args.save_preds = ''
                print('NEW FILE', args.filename)
                main(args)
            #plt.savefig('NoPerturbationErrorAnalysis.eps', format='eps')
            plt.show()
        else:
            main(args)
     
