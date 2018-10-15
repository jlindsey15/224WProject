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


os.environ["CUDA_VISIBLE_DEVICES"] = "1,"

#os.chdir('/home/druckmannuser/NeuralPopulationAnalysisUtils/Code/Python')

from models import Simple
#from lfads_datasets import DualALMSimpleDataset
os.chdir('..')
cwd = os.getcwd()

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--batch-size', type=int, default=2, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=20, metavar='N',
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
parser.add_argument('--trainstimtype', default=None, help='no stim, early delay left ALM, early delay right ALM, early delay bi ALM')
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
parser.add_argument('--timebins', default='', type=str)
parser.add_argument('--use_exp', action='store_true', default=False)
parser.add_argument('--no_cuda', action='store_true', default=False)
parser.add_argument('--save_preds', default=cwd+'/SavedPreds/', type=str, help='Destination path for trained model preds')
parser.add_argument('--neuron_split', type=float, default=1.0, help='Fraction of neurons to train on')
parser.add_argument('--f', type=int, default=10, help='Number of latent factors')

args = parser.parse_args()

'''
if args.bridge == 'full' or args.bridge =='' or args.bridge == 'scalargatesym' or args.bridge =='scalarmultsym' or args.bridge=='scalargatemixsym' or args.bridge=='scalargatefxsym' or args.bridge=='scalargateCDsym' or args.bridge=='scalargateadjsym' or args.bridge=='dual' or args.bridge == 'dualscalargatesym' or args.bridge =='scalargatesymFromL' or args.bridge == 'scalargatesymFromR' or args.bridge == 'scalargateFromSource' or args.bridge == 'scalargateFromTarget' or args.bridge == 'scalargateFromTargetCD' or args.bridge == 'scalargateFromPredTargetCD':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,"
if args.bridge == 'all_all' or args.bridge == 'all_CD' or args.bridge == 'linear' or args.bridge == 'scalargatemult' or args.bridge =='scalargate' or args.bridge=='scalargatemix' or args.bridge=='scalargatefx' or args.bridge=='scalargateCD' or args.bridge=='scalargateadj' or args.bridge=='dualscalargate' or args.bridge== 'scalargatesymFromSource' or args.bridge == 'scalargatesymFromTarget' or args.bridge == 'scalargateFromL' or args.bridge == 'scalargateFromR' or args.bridge == 'scalargateFromBothCD' or args.bridge == 'scalargateFromPredBothCD':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1,"
if args.bridge == 'to_CD' or args.bridge == 'from_CD' or args.bridge == 'scalargateX' or args.bridge == 'scalargatemultsymX':
    os.environ["CUDA_VISIBLE_DEVICES"] = "2,"
'''
if 'attractor' in args.bridge:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,"
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = "1,"

args.cuda = not args.no_cuda and torch.cuda.is_available()

filenames = ['Data/DualALMRecordinbgsDataSet2018_04_14/BAYLORGC4/BAYLORGC4_2018_03_22.mat', 'Data/DualALMRecordinbgsDataSet2018_04_14/BAYLORGC4/BAYLORGC4_2018_03_21.mat', 'Data/DualALMRecordinbgsDataSet2018_04_14/BAYLORGC4/BAYLORGC4_2018_03_23.mat', 'Data/DualALMRecordinbgsDataSet2018_04_14/BAYLORGC4/BAYLORGC4_2018_03_20.mat', 'Data/DualALMRecordinbgsDataSet2018_04_14/BAYLORGC4/BAYLORGC4_2018_03_25.mat', 'Data/DualALMRecordinbgsDataSet2018_04_14/BAYLORGC12/BAYLORGC12_2018_04_03.mat', 'Data/DualALMRecordinbgsDataSet2018_04_14/BAYLORGC12/BAYLORGC12_2018_04_04.mat', 'Data/DualALMRecordinbgsDataSet2018_04_14/BAYLORGC12/BAYLORGC12_2018_04_05.mat', 'Data/DualALMRecordinbgsDataSet2018_04_14/BAYLORGC12/BAYLORGC12_2018_04_06.mat', 'Data/DualALMRecordinbgsDataSet2018_04_14/BAYLORGC13/BAYLORGC13_2018_04_23.mat', 'Data/DualALMRecordinbgsDataSet2018_04_14/BAYLORGC13/BAYLORGC13_2018_04_24.mat', 'Data/DualALMRecordinbgsDataSet2018_04_14/BAYLORGC13/BAYLORGC13_2018_04_25.mat', 'Data/DualALMRecordinbgsDataSet2018_04_14/BAYLORGC15/BAYLORGC15_2018_05_07.mat', 'Data/DualALMRecordinbgsDataSet2018_04_14/BAYLORGC15/BAYLORGC15_2018_05_08.mat', 'Data/DualALMRecordinbgsDataSet2018_04_14/BAYLORGC15/BAYLORGC15_2018_05_09.mat', 'Data/DualALMRecordinbgsDataSet2018_04_14/BAYLORGC15/BAYLORGC15_2018_05_10.mat', 'Data/DualALMRecordinbgsDataSet2018_04_14/BAYLORGC15/BAYLORGC15_2018_05_11.mat', 'Data/DualALMRecordinbgsDataSet2018_04_14/BAYLORGC17/BAYLORGC17_2018_06_08.mat', 'Data/DualALMRecordinbgsDataSet2018_04_14/BAYLORGC17/BAYLORGC17_2018_06_11.mat', 'Data/DualALMRecordinbgsDataSet2018_04_14/BAYLORGC18/BAYLORGC18_2018_05_31.mat', 'Data/DualALMRecordinbgsDataSet2018_04_14/BAYLORGC19/BAYLORGC19_2018_06_20.mat', 'Data/DualALMRecordinbgsDataSet2018_04_14/BAYLORGC19/BAYLORGC19_2018_06_21.mat', 'Data/DualALMRecordinbgsDataSet2018_04_14/BAYLORGC19/BAYLORGC19_2018_06_22.mat']

savenames = ['BAYLORGC4_2018_03_22.mat', 'BAYLORGC4_2018_03_21.mat', 'BAYLORGC4_2018_03_23.mat', 'BAYLORGC4_2018_03_20.mat', 'BAYLORGC4_2018_03_25.mat', 'BAYLORGC12_2018_04_03.mat', 'BAYLORGC12_2018_04_04.mat', 'BAYLORGC12_2018_04_05.mat', 'BAYLORGC12_2018_04_06.mat', 'BAYLORGC13_2018_04_23.mat', 'BAYLORGC13_2018_04_24.mat', 'BAYLORGC13_2018_04_25.mat', 'BAYLORGC15_2018_05_07.mat', 'BAYLORGC15_2018_05_08.mat', 'BAYLORGC15_2018_05_09.mat', 'BAYLORGC15_2018_05_10.mat', 'BAYLORGC15_2018_05_11.mat', 'BAYLORGC17_2018_06_08.mat', 'BAYLORGC17_2018_06_11.mat', 'BAYLORGC18_2018_05_31.mat', 'BAYLORGC19_2018_06_20.mat', 'BAYLORGC19_2018_06_21.mat', 'BAYLORGC19_2018_06_22.mat']

epoch_dict = {'presample': range(0, 9), 'sample': range(8, 21), 'presamplesample': range(0, 21), 'delay': range(20, 37), 'sampledelay': range(8, 37), 'task': range(8, 55), 'perturbation': range(24, 28), 'preduringperturbation': range(0, 26), 'postperturbation': range(27, 37), 'presamplesampledelay': range(0, 37)}




def main(args):
    trainstimtype = args.trainstimtype
    if args.trainstimtype is None:
        trainstimtype = ''


    args.save_preds += 'OneStepModel' + str(args.neuron_split) + '_' + str(args.bridge) + '_' + str(args.timebins)+str(trainstimtype)+str(args.weight_decay)+args.savename + '_'

    if args.resume == 'Default':
        args.resume = 'OneStepModel' + str(args.neuron_split) + '_' + str(args.bridge) + '_' + str(args.timebins)+str(trainstimtype)+str(args.weight_decay)+args.savename + '_checkpoint.pth.tar'

    if args.test_trained_model == 'Default':
        args.test_trained_model = 'OneStepModel' + str(args.neuron_split) + '_' + str(args.bridge) + '_' + str(args.timebins)+str(trainstimtype)+str(args.weight_decay)+ args.savename + '_checkpoint.pth.tar'

    if args.test_trained_model:
        if os.path.isfile(args.test_trained_model):
            print("=> loading checkpoint '{}'".format(args.test_trained_model))
            checkpoint = torch.load(args.test_trained_model, map_location=lambda storage, loc: storage)
            args.model = checkpoint['model']
            if 'location' in checkpoint:
                args.location = checkpoint['location']
            if 'model_size' in checkpoint:
                args.model_size = checkpoint['model_size']
        else:
            print("=> no checkpoint found at '{}'".format(args.test_trained_model))
        print('')

    if args.resume:
        print('Resume training from an existing model.')
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage)
            args.model = checkpoint['model']
            if 'location' in checkpoint:
                args.location = checkpoint['location']
            args.location = checkpoint['location']
            if 'model_size' in checkpoint:
                args.model_size = checkpoint['model_size']
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
        print('')

    print('CUDA enabled: {}'.format(args.cuda))
    if args.cuda:
        torch.cuda.set_device(args.gpu_idx)
        print('Current GPU Device: {}'.format(torch.cuda.current_device()))
    print('Model: {}'.format(args.model))
    if args.model_size:
        print('Model Size: {}'.format(args.model_size))
    print('Location: {}'.format(args.location))
    if args.test_trained_model:
        print('Trained Model Path: {}'.format(args.test_trained_model))
        print('Test a trained model.')
    else:
        print('Learning Rate: {:.1e}'.format(args.lr))
        print('Weight Decay: {:.1e}'.format(args.weight_decay))
        print('Batch Size: {}'.format(args.batch_size))
        print('Test Batch Size: {}'.format(args.test_batch_size))
        print('Epochs: {}'.format(args.epochs))


    #train_dataset = DualALMSimpleDataset(train=True, neuron_split=args.neuron_split, location=args.location, filename=args.filename, train_stim_type=args.trainstimtype, timebins=args.timebins, indexX = indexX)
    #test_dataset = DualALMSimpleDataset(train=False, neuron_split=args.neuron_split, location=args.location, filename=args.filename, train_stim_type=args.trainstimtype, timebins=args.timebins, indexX = indexX)

    filehandler = open(cwd+'/OneStepPreprocessed/'+args.savename+'ALLEVERYTHING_train_dataset.obj', 'rb')
    train_dataset = pickle.load(filehandler)
    filehandler = open(cwd+'/OneStepPreprocessed/'+args.savename+'ALLEVERYTHING_test_dataset.obj', 'rb')
    test_dataset = pickle.load(filehandler)


    kwargs = {'num_workers': 8, 'pin_memory': True} if args.cuda else {}
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)
    train_loader_no_shuffle = torch.utils.data.DataLoader(train_dataset, batch_size=args.test_batch_size, shuffle=False, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False, **kwargs)
    print('hiii', train_dataset.data.shape)
    data_L = train_dataset.data[:, np.array(train_dataset.pa.train_data.neural_unit_location=='left_ALM')]
    data_R = train_dataset.data[:, np.array(train_dataset.pa.train_data.neural_unit_location=='right_ALM')]

    CD_L = np.squeeze(train_dataset.cd_L)#train_dataset.coding_directions[36][np.array(train_dataset.pa.train_data.neural_unit_location=='left_ALM')]
    #CD_L = CD_L / np.linalg.norm(CD_L)
    CD_L = torch.cuda.FloatTensor(CD_L)
    CD_R = np.squeeze(train_dataset.cd_R)#train_dataset.coding_directions[36][np.array(train_dataset.pa.train_data.neural_unit_location=='right_ALM')]
    #CD_R = CD_R / np.linalg.norm(CD_R)
    CD_R = torch.cuda.FloatTensor(CD_R)

    if args.model_size == 'size1':
        model_size = {'input_size_L':data_L.shape[1], 'input_size_R':data_R.shape[1], 'g_enc_size':150, 'c_enc_size':100, 'ctr_size':128, 'gen_size':128, 'u_size':50, 'f_size':args.f, 'bridge':args.bridge, 'no_f':args.no_f, 'CD_L':CD_L, 'CD_R':CD_R, 'nocdsource':args.nocdsource}
    if args.non_variational:
        model_size['non_variational'] = True
    if args.linear:
        model_size['linear'] = True
    if args.superlinear:
        model_size['superlinear'] = True
    if args.duperlinear:
        model_size['duperlinear'] = True
    if args.no_ext_inp:
        model_size['no_ext_inp'] = True
    if args.provide_trial_type:
        model_size['provide_trial_type'] = True
    if args.use_exp:
        model_size['use_exp'] = True
    if args.test_trained_model:
        model_size['output_gate'] = True

    model_size['f2r_nonlins'] = args.f2r_nonlins

    if args.model == 'lfads':
        model = Simple(**model_size).double()

    

    if args.test_trained_model or args.resume:
        model.load_state_dict(checkpoint['state_dict'], strict=False)

    if args.cuda:
        model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # If size_average is True, it divides the sum of squared differences of all elements by the total number of elements.
    # For example, if the two tensors to compare have shape (seq_len, batch_size, input_size), it divides the total sum by
    # seq_len*batch_size*input_size. If size_average+False, it returns the total sum instead.
    reconstruction_function = nn.MSELoss(size_average=True)

    def loss_function(recon_x, x):

        x = np.squeeze(x.float())
        recon_x = np.squeeze(recon_x.float())
        recon_loss = reconstruction_function(recon_x, x)


        return recon_loss


    def lr_scheduler(optimizer, epoch, init_lr=0.001, lr_decay_epoch=10):
        """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
        lr = init_lr * (0.1**(epoch // lr_decay_epoch))

        if epoch > 0 and epoch % lr_decay_epoch == 0:
            print('LR is set to {}'.format(lr))
            print('')

            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        return optimizer

    def lr_scheduler2(optimizer, epoch, init_lr=0.001, lr_decay_epochs=[4,7]):
        for i, lr_decay_epoch in enumerate(lr_decay_epochs):
            if epoch == lr_decay_epoch:
                lr = init_lr * (0.1**(i+1))
                print('LR is set to {}'.format(lr))
                print('')

                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr

        return optimizer

    def train(epoch, optimizer):
        model.train()
        train_loss_L = 0.0
        train_loss_eval_L = 0.0
        train_loss_hist_L = []
        train_loss_R = 0.0
        train_loss_eval_R = 0.0
        train_loss_hist_R = []
        start_time = time.time()

        #optimizer = lr_scheduler(optimizer, epoch, init_lr=args.lr)
        for batch_idx, (data, label, linreg_L, linreg_R, nobridge_linreg_L, nobridge_linreg_R) in enumerate(train_loader):
            #print('yoooo', np.array(train_dataset.pa.train_data.neural_unit_location=='left_ALM'), data.shape)
            data_L = torch.squeeze(data[:, torch.Tensor((train_dataset.pa.train_data.neural_unit_location=='left_ALM').astype(int)).long().nonzero()], dim=2)
            data_R = torch.squeeze(data[:, torch.Tensor((train_dataset.pa.train_data.neural_unit_location=='right_ALM').astype(int)).long().nonzero()], dim=2)

            label_L = label[:, 0:1]
            label_R = label[:, 1:2]
            #print('blabadeebla', trial_type.shape)
            # Note that data has shape (batch_size, seq_len, input_size).
            if args.cuda:
                data_L = data_L.cuda()
                data_R = data_R.cuda()
                label_L = label_L.cuda()
                label_R = label_R.cuda()
                linreg_L = linreg_L.cuda()
                linreg_R = linreg_R.cuda()
                nobridge_linreg_L = nobridge_linreg_L.cuda()
                nobridge_linreg_R = nobridge_linreg_R.cuda()
            data_L = Variable(data_L)
            data_R = Variable(data_R)
            label_L = Variable(label_L)
            label_R = Variable(label_R)
            optimizer.zero_grad()

            recon_L, recon_R, _, _ = model([data_L, data_R, nobridge_linreg_L, nobridge_linreg_R])

            # loss is divided by batch_size so that it's always per each example, regardless of batch_size.
            loss_L = loss_function(recon_L, label_L)
            loss_R = loss_function(recon_R, label_R)
            loss = loss_L + loss_R
            loss.backward()

            optimizer.step()

            train_loss_L += loss_L.data[0]

            train_loss_R += loss_R.data[0]


            if (batch_idx+1) % args.log_interval == 0:
                duration = time.time() - start_time
                #print('LEFT Train Epoch: {} [{}/{} ({:.0f}%)]\tRecon Loss: {:.4e} ({:.3f} sec)'.format(
                #    epoch, (batch_idx+1)*len(data), len(train_loader.dataset),
                #    100.*(batch_idx+1)/len(train_loader), loss_L.data[0], duration))
                #print('RIGHT Train Epoch: {} [{}/{} ({:.0f}%)]\tRecon Loss: {:.4e} ({:.3f} sec)'.format(
                #    epoch, (batch_idx+1)*len(data), len(train_loader.dataset),
                #    100.*(batch_idx+1)/len(train_loader), loss_R.data[0], duration))
                start_time = time.time()


        train_loss_L /= len(train_loader)

        train_loss_R /= len(train_loader)

        print('\nLEFT Train set:  Average loss: {:.4e}\n'.format(train_loss_L))
        print('\nRIGHT Train set: Average loss: {:.4e}\n'.format(train_loss_R))

        return [train_loss_hist_L, train_loss_hist_R]

    def test(epoch, best_loss):
        model.eval()
        test_loss_L = 0.0
        test_loss_eval_L = 0.0
        test_loss_R = 0.0
        test_loss_eval_R = 0.0
        for data, label, linreg_L, linreg_R, nobridge_linreg_L, nobridge_linreg_R in test_loader:

            data_L = torch.squeeze(data[:, torch.Tensor((test_dataset.pa.test_data.neural_unit_location=='left_ALM').astype(int)).long().nonzero()], dim=2)
            data_R = torch.squeeze(data[:, torch.Tensor((test_dataset.pa.test_data.neural_unit_location=='right_ALM').astype(int)).long().nonzero()], dim=2)

            label_L = label[:, 0:1]
            label_R = label[:, 1:2]
            if args.cuda:
                data_L = data_L.cuda()
                data_R = data_R.cuda()
                label_L = label_L.cuda()
                label_R = label_R.cuda()
                linreg_L = linreg_L.cuda()
                linreg_R = linreg_R.cuda()
                nobridge_linreg_L = nobridge_linreg_L.cuda()
                nobridge_linreg_R = nobridge_linreg_R.cuda()
            data_L = Variable(data_L, volatile=True)
            data_R = Variable(data_R, volatile=True)
            label_L = Variable(label_L, volatile=True)
            label_R = Variable(label_R, volatile=True)
            if args.model == 'lfads':
                recon_L, recon_R, _, _ = model([data_L, data_R, nobridge_linreg_L, nobridge_linreg_R])
            if args.model == 'lfads':
                # loss is divided by batch_size so that it's always per each example, regardless of batch_size.
                loss_L = loss_function(recon_L, label_L)
                loss_R = loss_function(recon_R, label_R)
                #loss /= len(data)
                #recon_loss /= len(data)
            test_loss_L += loss_L.data[0]
            test_loss_R += loss_R.data[0]
        test_loss_L /= len(test_loader)
        test_loss_R /= len(test_loader)
        test_loss = test_loss_L + test_loss_R
        print('\nLEFT Test set: Average loss: {:.4e}\n'.format(
            test_loss_L))
        print('\nRIGHT Test set:  Average loss: {:.4e}\n'.format(
            test_loss_R))
        #best_loss = best_loss.cuda()
        #test_loss = test_loss.cuda()
        is_best = test_loss < best_loss
        if is_best:
            print('New best test loss achieved!')
        best_loss = min(test_loss, best_loss)
        if args.save_model:
            print('Saving the model...')
            state = vars(args)
            state['state_dict'] = model.state_dict()
            state['optimizer'] = optimizer.state_dict()
            state['best_loss'] = best_loss
            state['epoch'] = epoch
            save_checkpoint(state, is_best)
            print('Saved the model.')
            print('')
        return test_loss, best_loss

    def test_only():
        model.eval()
        test_loss_L = 0.0
        test_loss_eval_L = 0.0
        test_loss_R = 0.0
        test_loss_eval_R = 0.0
        print('Test loader length', len(test_loader))

        recons_L = []
        recons_R = []
        datas_L = []
        datas_R = []
        baselines_L = []
        baselines_R = []
        L2Rgates = []
        R2Lgates = []
        for data, label, linreg_L, linreg_R, nobridge_linreg_L, nobridge_linreg_R in test_loader:

            data_L = torch.squeeze(data[:, torch.Tensor((test_dataset.pa.test_data.neural_unit_location=='left_ALM').astype(int)).long().nonzero()], dim=2)
            data_R = torch.squeeze(data[:, torch.Tensor((test_dataset.pa.test_data.neural_unit_location=='right_ALM').astype(int)).long().nonzero()], dim=2)
            data_L_proj = torch.matmul(data_L, CD_L.cpu().double())
            data_R_proj = torch.matmul(data_R, CD_R.cpu().double())
            label_L = label[:, 0:1]
            label_R = label[:, 1:2]
            if args.cuda:
                data_L = data_L.cuda()
                data_R = data_R.cuda()
                label_L = label_L.cuda()
                label_R = label_R.cuda()
                linreg_L = linreg_L.cuda()
                linreg_R = linreg_R.cuda()
                nobridge_linreg_L = nobridge_linreg_L.cuda()
                nobridge_linreg_R = nobridge_linreg_R.cuda()
            data_L = Variable(data_L, volatile=True)
            data_R = Variable(data_R, volatile=True)
            label_L = Variable(label_L, volatile=True)
            label_R = Variable(label_R, volatile=True)
            if args.model == 'lfads':
                recon_L, recon_R, L2Rgate, R2Lgate = model([data_L, data_R, nobridge_linreg_L, nobridge_linreg_R])

                recons_L.append(recon_L.cpu().data.numpy())
                recons_R.append(recon_R.cpu().data.numpy())
                baselines_L.append(data_L_proj.cpu().data.numpy())
                baselines_R.append(data_R_proj.cpu().data.numpy())           
                datas_L.append(label_L.cpu().data.numpy())
                datas_R.append(label_R.cpu().data.numpy())
                L2Rgates.append(L2Rgate.cpu().data.numpy())
                R2Lgates.append(R2Lgate.cpu().data.numpy())
            if args.model == 'lfads':
                # loss is divided by batch_size so that it's always per each example, regardless of batch_size.
                loss_L = loss_function(recon_L, label_L)
                loss_R = loss_function(recon_R, label_R)
                #loss /= len(data)
                #recon_loss /= len(data)
            test_loss_L += loss_L.data[0]
            test_loss_R += loss_R.data[0]
        test_loss_L /= len(test_loader)
        test_loss_R /= len(test_loader)
        recons_L = np.array(recons_L)
        recons_R = np.array(recons_R)
        datas_L = np.array(datas_L)
        datas_R = np.array(datas_R)
        baselines_L = np.array(baselines_L)
        baselines_R = np.array(baselines_R)
        L2Rgates = np.array(L2Rgates)
        R2Lgates = np.array(R2Lgates)
        if args.save_preds:
            print('where to save', args.save_preds)
            np.save(args.save_preds+'TEST_L.npy', np.squeeze(recons_L))
            np.save(args.save_preds+'TEST_R.npy', np.squeeze(recons_R))
            np.save(args.save_preds+'TEST_TRUTH_L.npy', np.squeeze(datas_L))
            np.save(args.save_preds+'TEST_TRUTH_R.npy', np.squeeze(datas_R))
            np.save(args.save_preds+'TEST_BASELINE_L.npy', np.squeeze(baselines_L))
            np.save(args.save_preds+'TEST_BASELINE_R.npy', np.squeeze(baselines_R))
            np.save(args.save_preds+'TEST_L2Rgates.npy', np.squeeze(L2Rgates))
            np.save(args.save_preds+'TEST_R2Lgates.npy', np.squeeze(R2Lgates))
            np.save(args.save_preds+'TRAIN_stimtypes.npy', train_dataset.pa.train_data.stim_site)
            np.save(args.save_preds+'TEST_stimtypes.npy', test_dataset.pa.test_data.stim_site)
            np.save(args.save_preds+'L2Rgate_weight.npy', model.L2Rgate.weight.data)
            np.save(args.save_preds+'R2Lgate_weight.npy', model.R2Lgate.weight.data)
            np.save(args.save_preds+'L2R1_weight.npy', model.L2R1.weight.data)
            np.save(args.save_preds+'R2L1_weight.npy', model.R2L1.weight.data)
        print('\nLEFT Test set: Average loss: {:.4e}\n'.format(
            test_loss_L))
        print('\nRIGHT Test set: Average loss: {:.4e}\n'.format(
            test_loss_R))

        if args.save_preds:
            train_loss_L = 0.0
            train_loss_eval_L = 0.0
            train_loss_R = 0.0
            train_loss_eval_R = 0.0
            print('Train loader length', len(train_loader_no_shuffle))
            recons_L = []
            recons_R = []
            datas_L = []
            datas_R = []
            baselines_L = []
            baselines_R = []
            L2Rgates = []
            R2Lgates = []
            for data, label, linreg_L, linreg_R, nobridge_linreg_L, nobridge_linreg_R in train_loader_no_shuffle:

                data_L = torch.squeeze(data[:, torch.Tensor((train_dataset.pa.train_data.neural_unit_location=='left_ALM').astype(int)).long().nonzero()], dim=2)
                data_R = torch.squeeze(data[:, torch.Tensor((train_dataset.pa.train_data.neural_unit_location=='right_ALM').astype(int)).long().nonzero()], dim=2)
                data_L_proj = torch.matmul(data_L, CD_L.cpu().double())
                data_R_proj = torch.matmul(data_R, CD_R.cpu().double())
                label_L = label[:, 0:1]
                label_R = label[:, 1:2]
                if args.cuda:
                    data_L = data_L.cuda()
                    data_R = data_R.cuda()
                    label_L = label_L.cuda()
                    label_R = label_R.cuda()
                    linreg_L = linreg_L.cuda()
                    linreg_R = linreg_R.cuda()
                    nobridge_linreg_L = nobridge_linreg_L.cuda()
                    nobridge_linreg_R = nobridge_linreg_R.cuda()
                data_L = Variable(data_L, volatile=True)
                data_R = Variable(data_R, volatile=True)
                label_L = Variable(label_L, volatile=True)
                label_R = Variable(label_R, volatile=True)
                if args.model == 'lfads':
                    recon_L, recon_R, L2Rgate, R2Lgate = model([data_L, data_R, nobridge_linreg_L, nobridge_linreg_R])

                    recons_L.append(recon_L.cpu().data.numpy())
                    recons_R.append(recon_R.cpu().data.numpy())
                    datas_L.append(label_L.cpu().data.numpy())
                    datas_R.append(label_R.cpu().data.numpy())
                    baselines_L.append(data_L_proj.cpu().data.numpy())
                    baselines_R.append(data_R_proj.cpu().data.numpy())
                    L2Rgates.append(L2Rgate.cpu().data.numpy())
                    R2Lgates.append(R2Lgate.cpu().data.numpy())
                if args.model == 'lfads':
                    # loss is divided by batch_size so that it's always per each example, regardless of batch_size.
                    loss_L = loss_function(recon_L, label_L)
                    loss_R = loss_function(recon_R, label_R)
                    #loss /= len(data)
                    #recon_loss /= len(data)
                train_loss_L += loss_L.data[0]

                train_loss_R += loss_R.data[0]

            train_loss_L /= len(train_loader_no_shuffle)
            train_loss_R /= len(train_loader_no_shuffle)
            recons_L = np.array(recons_L)
            recons_R = np.array(recons_R)
            baselines_L = np.array(baselines_L)
            baselines_R = np.array(baselines_R)
            datas_L = np.array(datas_L)
            datas_R = np.array(datas_R)
            L2Rgates = np.array(L2Rgates)
            R2Lgates = np.array(R2Lgates)
            if args.save_preds:
                np.save(args.save_preds+'TRAIN_L.npy', np.squeeze(recons_L))
                np.save(args.save_preds+'TRAIN_R.npy', np.squeeze(recons_R))
                np.save(args.save_preds+'TRAIN_TRUTH_L.npy', np.squeeze(datas_L))
                np.save(args.save_preds+'TRAIN_TRUTH_R.npy', np.squeeze(datas_R))
                np.save(args.save_preds+'TRAIN_BASELINE_L.npy', np.squeeze(baselines_L))
                np.save(args.save_preds+'TRAIN_BASELINE_R.npy', np.squeeze(baselines_R))
                np.save(args.save_preds+'TRAIN_L2Rgates.npy', np.squeeze(L2Rgates))
                np.save(args.save_preds+'TRAIN_R2Lgates.npy', np.squeeze(R2Lgates))
            print('\nLEFT Train set:  Average loss: {:.4e}\n'.format(train_loss_L))
            print('\nRIGHT Train set: Average loss: {:.4e}\n'.format(train_loss_R))

        else:

            start = min(0, len(data))
            n_visualize = min(20, len(data)-start)

            '''
            for i in range(start, start + n_visualize):
                vmin = min(np.amin(data[i].data.cpu().numpy()), np.amin(recon[i].data.cpu().numpy()))
                vmax = max(np.amax(data[i].data.cpu().numpy()), np.amax(recon[i].data.cpu().numpy()))
                print('')
                print('<{}-th plot>'.format(i+1))
                print('vmin:{}'.format(vmin))
                print('vmax:{}'.format(vmax))
                gs = gridspec.GridSpec(4,5)
                ax = plt.subplot(gs[:,:2])
                ax.set_title('Original Data')
                ax.set_xlabel('time points')
                ax.set_ylabel('neurons')
                plt.imshow(np.transpose(data[i].data.cpu().numpy()), vmin=vmin, vmax=vmax)
                ax = plt.subplot(gs[:,2:4])
                ax.set_title('Reconstructed Data')
                ax.set_xlabel('time points')
                ax.set_ylabel('neurons')
                recon_im = plt.imshow(np.transpose(recon[i].data.cpu().numpy()), vmin=vmin, vmax=vmax)
                plt.colorbar(recon_im, cax=plt.subplot(gs[1:3,4]))
                plt.tight_layout()
                #plt.show()
            '''
    def save_checkpoint(state, is_best):

        model_name = 'OneStepModel' + str(args.neuron_split) + '_' + str(args.bridge) + '_' + str(args.timebins)+str(trainstimtype)+str(args.weight_decay)+args.savename
        file_name = model_name + '_checkpoint.pth.tar'        
        torch.save(state, file_name)
        if is_best:
            shutil.copyfile(file_name, model_name + '_best.pth.tar')

    if args.test_trained_model:
        test_only()
    else:
        if args.resume:
            best_loss = checkpoint['best_loss']
        else:
            best_loss = float('inf')
        train_loss_hist = []
        test_loss_hist = []
        for epoch in range(1, args.epochs + 1):
            # Note that len(train_loss_hist) = number of iterations,
            # And len(test_loss_hist) = number of epochs
            start_time = time.time()
            train_loss_hist += train(epoch, optimizer)
            test_loss, best_loss = test(epoch, best_loss)
            test_loss_hist.append(test_loss)
            duration = time.time() - start_time
            print('Total time taken for Epoch {}: {:.3f} sec'.format(epoch, duration))
            print('')
        test_only()
        return train_loss_hist, test_loss_hist

def hyperparameter_search(args):
    print('')
    print('Hyperparemter Search.')
    print('')
    assert not args.test_trained_model
    assert not args.save_preds
    args.epochs = 3
    search_lr = False
    search_batch_size = False
    search_weight_decay = True

    # For lr search, we need to look at the train loss curve.
    if search_lr:
        lrs = [10**n for n in range(-4,-1)]
        original_weight_decay = args.weight_decay
        args.weight_decay = 0.0
        original_batch_size = args.batch_size
        args.batch_size = 100
        for lr in lrs:
            args.lr = lr
            train_loss_hist, _ = main(args)
            plt.plot(train_loss_hist, label='{:.0e}'.format(args.lr))
        plt.title('Train loss history')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.legend(ncol=len(lrs), loc='lower right')
        #plt.show()
        args.weight_decay = original_weight_decay
        args.batch_size = original_batch_size
    args.lr = 1e-3 # the best value found above.

    # For batch_size search, both the train (how noisy is the gradient descent?) and test (how well the model generalizes?) loss curve matters.
    if search_batch_size:
        batch_sizes = [25, 50, 100]
        testset_size = 10000
        original_weight_decay = args.weight_decay
        args.weight_decay = 0.0
        original_epochs = args.epochs
        for batch_size in batch_sizes:
            args.batch_size = batch_size
            # To keep the total number of iterations same over different batch sizes
            args.epochs = original_epochs*batch_size/batch_sizes[0]
            train_loss_hist, test_loss_hist = main(args)
            plt.subplot(2,1,1)
            plt.plot(train_loss_hist, label='{}'.format(args.batch_size))
            plt.subplot(2,1,2)
            plt.plot([testset_size/batch_size*(i+1) for i in range(args.epochs)], test_loss_hist, label='{}'.format(args.batch_size))
        plt.subplot(2,1,1)
        plt.title('Train loss history')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.legend(ncol=len(batch_sizes), loc='lower right')
        plt.subplot(2,1,2)
        plt.title('Test loss history')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.legend(ncol=len(batch_sizes), loc='lower right')
        plt.tight_layout()
        #plt.show()
        args.weight_decay = original_weight_decay
        args.epochs = original_epochs # restore the original epochs.
    args.batch_size = 50 # the best value found above.

    # For weight_decay, we need to look at test loss curve.
    if search_weight_decay:
        weight_decays = [0.0] + [10**n for n in range(-5, -1)]
        for weight_decay in weight_decays:
            args.weight_decay = weight_decay
            _, test_loss_hist = main(args)
            plt.plot(test_loss_hist, label='{:.0e}'.format(args.weight_decay))
        plt.title('Test loss history')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(ncol=len(weight_decays)//2, loc='lower right')
        #plt.show()            
    args.weight_decay = 0.0 # the best value found above.


if __name__ == '__main__':
    if args.hp_search:
        hyperparameter_search(args)
    else:
        if args.filename is None:

            for indexX in range(len(filenames)):
              for wd in [args.weight_decay]:
                args.filename = filenames[indexX]
                args.savename = savenames[indexX]
                args.weight_decay = wd
                #args.test_trained_model = 'Default'
                args.save_preds = cwd+'/SavedPreds/'
                print('NEW FILE', args.filename)
                main(args)
        else:
            main(args)
     
