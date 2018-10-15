import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.autograd import Variable
import numpy as np





class Simple(nn.Module):
    '''
    Note: in the LFADS paper, they mention in passing that they averaged a large number (e.g. 128) of samples of g_0 
    conditioned on the particular sequence. Presumably, this leads to a more accurate estimate of g_0 and therefore more 
    accurate reconstruction I guess. But, in this implementation, we only sample a single g_0 to generate the reconstruction,
    as usual for VAEs.
    '''
    def __init__(self, input_size_L, input_size_R, g_enc_size, c_enc_size, ctr_size, gen_size, u_size, f_size, non_variational=False, linear=False, no_ext_inp=False, superlinear=False, duperlinear=False, nbins=55, provide_trial_type=False, CD_L = None, CD_R=None, no_f='', use_exp=False, bridge="", f2r_nonlins=0, source_bins=range(10), target_bins=range(55), sample_bins=range(8, 21), nocdsource='', just_observed = True, output_gate = False):
        super(Simple, self).__init__()


        self.relu = torch.nn.ReLU()
        self.tanh = torch.nn.Tanh()
        def nonlinFunc(x):
            if 'Mono' in bridge:
                x = self.relu(x)
            x = torch.abs(x)
            if 'Linear' in bridge:
                y = x
            elif 'Square' in bridge:
                y = torch.pow(x, 2)
            elif 'Root' in bridge:
                y = torch.pow(x, 0.5)
            if 'Tanh' in bridge:
                y = self.tanh(y)
            y = torch.clamp(y, min=0.0, max=1.0)
            if 'Flip' in bridge:
                y = 1 - y
            return y
        self.nonlinearity = nonlinFunc
        self.bridge = bridge
        self.output_gate = output_gate
        self.L2LFull = nn.Linear(input_size_L, input_size_L)
        self.R2LFull = nn.Linear(input_size_R, input_size_L)
        self.L2RFull = nn.Linear(input_size_L, input_size_R)
        self.R2RFull = nn.Linear(input_size_R, input_size_R)
        self.L2LCD = nn.Linear(input_size_L, 1)
        self.R2RCD = nn.Linear(input_size_R, 1)
        self.L2L1 = nn.Linear(input_size_L, 20)
        self.L2L2 = nn.Linear(20, 1)
        self.R2R1 = nn.Linear(input_size_R, 20)
        self.R2R2 = nn.Linear(20, 1)

        self.L2LA = nn.Linear(1, 1)
        self.R2RA = nn.Linear(1, 1)
        self.L2LB = nn.Linear(1, 1)
        self.R2RB = nn.Linear(1, 1)
        self.L2LDualGate = nn.Linear(1, 1)
        self.R2RDualGate = nn.Linear(1, 1)
        self.L2R1 = nn.Linear(input_size_L, 1)
        self.L2R2 = nn.Linear(1, input_size_R)
        self.R2L1 = nn.Linear(input_size_R, 1)
        self.R2L2 = nn.Linear(1, input_size_L)
        #self.L2RCD = nn.Linear(1, 1)
        #self.R2LCD = nn.Linear(1, 1)
        self.AL = nn.Linear(input_size_L+input_size_R, 1)
        self.AR = nn.Linear(input_size_L+input_size_R, 1)
        self.BL = nn.Linear(input_size_L+input_size_R, 1)
        self.BR = nn.Linear(input_size_L+input_size_R, 1)
        self.L2Rgate = nn.Linear(input_size_L+input_size_R, 1)
        self.R2Lgate = nn.Linear(input_size_L+input_size_R, 1)
        self.L2RgateCD = nn.Linear(1, 1)
        self.R2LgateCD = nn.Linear(1, 1)
        self.L2RgateBothCD = nn.Linear(2, 1)
        self.R2LgateBothCD = nn.Linear(2, 1)
        self.L2RgateFromL = nn.Linear(input_size_L, 1)
        self.R2LgateFromL = nn.Linear(input_size_L, 1)
        self.L2RgateFromR = nn.Linear(input_size_R, 1)
        self.R2LgateFromR = nn.Linear(input_size_R, 1)
        self.L2RgateVec = nn.Linear(input_size_L+input_size_R, input_size_R)
        self.R2LgateVec = nn.Linear(input_size_L+input_size_R, input_size_L)
        #print('CDLsize', CD_L.size())
        self.CD_L = CD_L
        self.CD_R = CD_R
        self.L2LCD.weight.data.copy_(CD_L)
        self.R2RCD.weight.data.copy_(CD_R)
        self.L2LCD.bias.data.fill_(0)#copy_(CD_L)
        self.R2RCD.bias.data.fill_(0)#copy_(CD_R)
        self.R2Lmix = nn.Linear(input_size_R, 1)
        self.L2Rmix = nn.Linear(input_size_L, 1)
        self.attractor1_L = nn.Linear(1, 1)
        self.attractor2_L = nn.Linear(1, 1)
        self.attractor1_R = nn.Linear(1, 1)
        self.attractor2_R = nn.Linear(1, 1)
        self.attractionRateL = nn.Linear(1, 1)
        self.attractionBalanceL = nn.Linear(input_size_L, 1)
        self.attractionRateR = nn.Linear(1, 1)
        self.attractionBalanceR = nn.Linear(input_size_R, 1)
        self.sig = torch.nn.Sigmoid()
        
    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = Variable(torch.FloatTensor(std.size()).normal_())
        if mu.is_cuda:
            eps = eps.cuda()
        return eps.mul(std).add_(mu)

    def forward(self, x):
        '''
        x: a tensor of shape (batch_size, seq_len, input_size)
        '''
        # rs is a list of r_t's. This is the reconstruction of x.
        x_L = x[0] #left hemisphere activity
        x_R = x[1] #right hemisphere activity
        linreg_L = x[2].view(-1, 1) #the within-hemisphere linear regression prediction on the left
        linreg_R = x[3].view(-1, 1) #the within-hemisphere linear regression prediction on the right
        x = torch.cat([x_L, x_R], dim=1)
        for p in self.L2LCD.parameters():
            p.requires_grad=False
        for p in self.R2RCD.parameters():
            p.requires_grad=False
        x_L_ortho = x_L - self.L2LCD(x_L) * self.CD_L.double()
        x_R_ortho = x_R - self.R2RCD(x_R) * self.CD_R.double()
        zerodummy = self.L2LCD(x_L) * 0
        if 'attractor' in self.bridge:

            attractor1_L = self.attractor1_L(zerodummy)
            attractor2_L = self.attractor2_L(zerodummy)
            attractor1_R = self.attractor1_R(zerodummy)
            attractor2_R = self.attractor2_R(zerodummy)

            self.attractionRateL.bias.data.fill_(torch.clamp(self.attractionRateL.bias.data[0], min=0.0, max=1.0))
            self.attractionRateR.bias.data.fill_(torch.clamp(self.attractionRateR.bias.data[0], min=0.0, max=1.0))
            attractionRateL = self.attractionRateL(zerodummy)
            attractionRateR = self.attractionRateR(zerodummy)
            attractionBalanceL = self.sig(self.attractionBalanceL(x_L))
            attractionBalanceR = self.sig(self.attractionBalanceR(x_R))
            attractor_L = (1 - attractionBalanceL) * attractor1_L + attractionBalanceL * attractor2_L
            attractor_R = (1 - attractionBalanceR) * attractor1_R + attractionBalanceR * attractor2_R
            linreg_L = (1 - attractionRateL) * linreg_L + attractionRateL * attractor_L
            linreg_R = (1 - attractionRateR) * linreg_R + attractionRateR * attractor_R


        if 'nobridge' in self.bridge: #no bridge
            pred_L = linreg_L
            pred_R = linreg_R
            return pred_L, pred_R, pred_L, pred_R
        elif 'normalbridge' in self.bridge: #ungated contralateral interactions
            pred_L = linreg_L + self.R2L1(x_R)
            pred_R = linreg_R + self.L2R1(x_L)
            return pred_L, pred_R, pred_L, pred_R
        elif 'scalargateFromTargetOrtho' in self.bridge: 
            #forces the gating factor to be a function of the target side activity orthogonalized against the CD
            pred_L = linreg_L + (self.nonlinearity(self.R2LgateFromL(x_L_ortho))) * (self.R2L1(x_R))
            pred_R = linreg_R + (self.nonlinearity(self.L2RgateFromR(x_R_ortho))) * (self.L2R1(x_L))

            return pred_L, pred_R, (self.nonlinearity(self.L2RgateFromR(x_R_ortho))), (self.nonlinearity(self.R2LgateFromL(x_L_ortho)))
        elif 'scalargateFromContraInf' in self.bridge:
            #control that forces the gating factor to be a function of the contralateral projection.  Gives a sense of much nonlinearity alone is helping the gating models
            pred_L = linreg_L + (self.nonlinearity(self.R2LgateCD(self.R2L1(x_R)))) * (self.R2L1(x_R))
            pred_R = linreg_R + (self.nonlinearity(self.L2RgateCD(self.L2R1(x_L)))) * (self.L2R1(x_L))
            return pred_L, pred_R, (self.nonlinearity(self.L2RgateCD(self.L2R1(x_L)))), (self.nonlinearity(self.R2LgateCD(self.R2L1(x_R))))#(pred_L - self.L2LFull(x_L)), (pred_R -
        elif 'scalargateFromAll' in self.bridge:
            #gating factor is a function of all activity
            pred_L = linreg_L + (self.nonlinearity(self.R2Lgate(x))) * (self.R2L1(x_R))
            pred_R = linreg_R + (self.nonlinearity(self.L2Rgate(x))) * (self.L2R1(x_L))
            return pred_L, pred_R, (self.nonlinearity(self.L2Rgate(x))), (self.nonlinearity(self.R2Lgate(x)))
        elif 'scalargateFromTargetCD' in self.bridge:
            #gating factor is a function of target side CD activity

            pred_L = linreg_L + (self.nonlinearity(self.R2LgateCD(self.L2LCD(x_L)))) * (self.R2L1(x_R))
            pred_R = linreg_R + (self.nonlinearity(self.L2RgateCD(self.R2RCD(x_R)))) * (self.L2R1(x_L))
            return pred_L, pred_R, (self.nonlinearity(self.L2RgateCD(self.R2RCD(x_R)))), (self.nonlinearity(self.R2LgateCD(self.L2LCD(x_L))))#(pred_L - self.L2LFull(x_L)), (pred_R -
        elif 'scalargateFromPredTargetCD' in self.bridge:
            #gating factor is a function of predicted target side CD activity
            pred_L = linreg_L + (self.nonlinearity(self.R2LgateCD(linreg_L))) * (self.R2L1(x_R))
            pred_R = linreg_R + (self.nonlinearity(self.L2RgateCD(linreg_R))) * (self.L2R1(x_L))
            return pred_L, pred_R, (self.nonlinearity(self.L2RgateCD(linreg_R))), (self.nonlinearity(self.R2LgateCD(linreg_L)))#(pred_L - self.L2LFull(x_L)), (pred_R -
        elif 'scalargateFromBothCD' in self.bridge:
            #gating factor is a function of both sides' CD activity
            both = torch.cat([self.L2LCD(x_L), self.R2RCD(x_R)], dim=1)
            pred_L = linreg_L + (self.nonlinearity(self.R2LgateBothCD(both))) * (self.R2L1(x_R))
            pred_R = linreg_R + (self.nonlinearity(self.L2RgateBothCD(both))) * (self.L2R1(x_L))
            return pred_L, pred_R, (self.nonlinearity(self.L2RgateBothCD(both))), (self.nonlinearity(self.R2LgateBothCD(both)))#(pred_L - self.L2LFull(x_L)), (pred_R
        elif 'scalargateFromPredBothCD' in self.bridge:
            #gating factor is a function of both sides' predicted CD activity
            predboth = torch.cat([linreg_L, linreg_R], dim=1)
            pred_L = linreg_L + (self.nonlinearity(self.R2LgateBothCD(predboth))) * (self.R2L1(x_R))
            pred_R = linreg_R + (self.nonlinearity(self.L2RgateBothCD(predboth))) * (self.L2R1(x_L))
            return pred_L, pred_R, (self.nonlinearity(self.L2RgateBothCD(predboth))), (self.nonlinearity(self.R2LgateBothCD(predboth)))#(pred_L - self.L2LFull(x_L)), (pred_R - self.R2RFull(x_R))
        elif 'scalargateFromLeft' in self.bridge:
            #gating factor is a function of left hemisphere activity
            pred_L = linreg_L + (self.nonlinearity(self.R2LgateFromL(x_L))) * (self.R2L1(x_R))
            pred_R = linreg_R + (self.nonlinearity(self.L2RgateFromL(x_L))) * (self.L2R1(x_L))
            return pred_L, pred_R, (self.nonlinearity(self.L2RgateFromL(x_L))), (self.nonlinearity(self.R2LgateFromL(x_L)))
        elif 'scalargateFromRight' in self.bridge:
            #gating factor is a function of right hemisphere activity
            pred_L = linreg_L + (self.nonlinearity(self.R2LgateFromR(x_R))) * (self.R2L1(x_R))
            pred_R = linreg_R + (self.nonlinearity(self.L2RgateFromR(x_R))) * (self.L2R1(x_L))
            return pred_L, pred_R, (self.nonlinearity(self.L2RgateFromR(x_R))), (self.nonlinearity(self.R2LgateFromR(x_R)))
        elif 'scalargateFromSource' in self.bridge:
            #gating factor is a function of source side activity
            pred_L = linreg_L + (self.nonlinearity(self.R2LgateFromR(x_R))) * (self.R2L1(x_R))
            pred_R = linreg_R + (self.nonlinearity(self.L2RgateFromL(x_L))) * (self.L2R1(x_L))
            return pred_L, pred_R, (self.nonlinearity(self.L2RgateFromL(x_L))), (self.nonlinearity(self.R2LgateFromR(x_R)))
        elif 'scalargateFromTarget' in self.bridge:
            #gating factor is a function of target side activity
            pred_L = linreg_L + (self.nonlinearity(self.R2LgateFromL(x_L))) * (self.R2L1(x_R))
            pred_R = linreg_R + (self.nonlinearity(self.L2RgateFromR(x_R))) * (self.L2R1(x_L))
            return pred_L, pred_R, (self.nonlinearity(self.L2RgateFromR(x_R))), (self.nonlinearity(self.R2LgateFromL(x_L)))




        #you can ignore the models below, they are either "experimental" or "abandoned"
        elif self.bridge == 'ff':
            pred_L = self.L2L2(self.nonlinearity(self.L2L1(x_L))) + linreg_L
            pred_R = self.R2R2(self.nonlinearity(self.R2R1(x_R))) + linreg_R
        elif self.bridge == 'dual':
            for p in self.L2L.parameters():
                p.requires_grad=False
            for p in self.R2R.parameters():
                p.requires_grad=False
            CD_proj_L = self.L2LCD(x_L)
            CD_proj_R = self.R2RCD(x_R)
            pred_L = (linreg_L + self.L2LA(CD_proj_L)) * self.nonlinearity(self.L2LDualGate(CD_proj_L)) + (linreg_L + self.L2LB(CD_proj_L)) * (1 - self.nonlinearity(self.L2LDualGate(CD_proj_L)))
            pred_R = (linreg_R + self.R2RA(CD_proj_R)) * self.nonlinearity(self.R2RDualGate(CD_proj_R)) + (linreg_R + self.R2RB(CD_proj_R)) * (1 - self.nonlinearity(self.R2RDualGate(CD_proj_R)))
        elif self.bridge == 'dualscalargate':
            for p in self.L2L.parameters():
                p.requires_grad=False
            for p in self.R2R.parameters():
                p.requires_grad=False
            CD_proj_L = self.L2LCD(x_L)
            CD_proj_R = self.R2RCD(x_R)
            pred_L = (linreg_L + self.L2LA(x_L)) * self.nonlinearity(self.L2LDualGate(CD_proj_L)) + (linreg_L + self.L2LB(x_L)) * (1 - self.nonlinearity(self.L2LDualGate(CD_proj_L))) + self.nonlinearity(self.R2Lgate(x)) * (self.R2L1(x_R))
            pred_R = (linreg_R + self.R2RA(x_R)) * self.nonlinearity(self.R2RDualGate(CD_proj_R)) + (linreg_R + self.R2RB(x_R)) * (1 - self.nonlinearity(self.R2RDualGate(CD_proj_R))) + self.nonlinearity(self.L2Rgate(x)) * (self.L2R1(x_L))

        elif self.bridge == 'dualscalargatesym':
            for p in self.L2L.parameters():
                p.requires_grad=False
            for p in self.R2R.parameters():
                p.requires_grad=False
            CD_proj_L = self.L2LCD(x_L)
            CD_proj_R = self.R2RCD(x_R)
            pred_L = (1 - self.nonlinearity(self.R2Lgate(x))) * ((linreg_L + self.L2LA(x_L)) * self.nonlinearity(self.L2LDualGate(CD_proj_L)) + (linreg_L + self.L2LB(x_L)) * (1 - self.nonlinearity(self.L2LDualGate(CD_proj_L)))) + self.nonlinearity(self.R2Lgate(x)) * (self.R2L1(x_R))
            pred_R = (1 - self.nonlinearity(self.L2Rgate(x))) * ((linreg_R + self.R2RA(x_R)) * self.nonlinearity(self.R2RDualGate(CD_proj_R)) + (linreg_R + self.R2RB(x_R)) * (1 - self.nonlinearity(self.R2RDualGate(CD_proj_R)))) + self.nonlinearity(self.L2Rgate(x)) * (self.L2R1(x_L))
        elif self.bridge == 'scalargateadj':
            pred_L = linreg_L + self.L2LCD(x_L) + self.nonlinearity(self.R2Lgate(x)) * (self.R2L1(x_R))
            pred_R = linreg_R + self.R2RCD(x_R) + self.nonlinearity(self.L2Rgate(x)) * (self.L2R1(x_L))
        elif self.bridge == 'scalargateCD':
            for p in self.L2L.parameters():
                p.requires_grad=False
            for p in self.R2R.parameters():
                p.requires_grad=False
            CD_proj_L = self.L2LCD(x_L)
            CD_proj_R = self.R2RCD(x_R)
            pred_L = linreg_L + self.nonlinearity(self.R2Lgate(x)) * (self.R2L1(CD_proj_R))
            pred_R = linreg_R + self.nonlinearity(self.L2Rgate(x)) * (self.L2R1(CD_proj_L))
        elif self.bridge == 'scalargatefx':
            pred_L = linreg_L + self.nonlinearity(self.R2Lgate(x)) * linreg_R
            pred_R = linreg_R + self.nonlinearity(self.L2Rgate(x)) * linreg_L
        elif self.bridge == 'scalargatemix':
            pred_L = linreg_L + self.nonlinearity(self.R2Lgate(x)) * (self.R2L1(x_R)) + self.R2Lmix(x_R)
            pred_R = linreg_R + self.nonlinearity(self.L2Rgate(x)) * (self.L2R1(x_L)) + self.L2Rmix(x_L)
        elif self.bridge == 'scalarnosig':
            pred_L = linreg_L + (self.R2Lgate(x)) * (self.R2L1(x_R))
            pred_R = linreg_R + (self.L2Rgate(x)) * (self.L2R1(x_L))
        elif self.bridge == 'scalargatesym':
            pred_L = (1-self.nonlinearity(self.R2Lgate(x))) * linreg_L + self.nonlinearity(self.R2Lgate(x)) * (self.R2L1(x_R))
            pred_R = (1 - self.nonlinearity(self.L2Rgate(x))) * linreg_R + self.nonlinearity(self.L2Rgate(x)) * (self.L2R1(x_L))
        elif self.bridge == 'scalargatesymFromL':
            pred_L = (1-self.nonlinearity(self.R2LgateFromL(x_L))) * linreg_L + self.nonlinearity(self.R2LgateFromL(x_L)) * (self.R2L1(x_R))
            pred_R = (1 - self.nonlinearity(self.L2RgateFromL(x_L))) * linreg_R + self.nonlinearity(self.L2RgateFromL(x_L)) * (self.L2R1(x_L))
        elif self.bridge == 'scalargatesymFromR':
            pred_L = (1-self.nonlinearity(self.R2LgateFromR(x_R))) * linreg_L + self.nonlinearity(self.R2LgateFromR(x_R)) * (self.R2L1(x_R))
            pred_R = (1 - self.nonlinearity(self.L2RgateFromR(x_R))) * linreg_R + self.nonlinearity(self.L2RgateFromR(x_R)) * (self.L2R1(x_L))
        elif self.bridge == 'scalargatesymFromSource':
            pred_L = (1-self.nonlinearity(self.R2LgateFromR(x_R))) * linreg_L + self.nonlinearity(self.R2LgateFromR(x_R)) * (self.R2L1(x_R))
            pred_R = (1 - self.nonlinearity(self.L2RgateFromL(x_L))) * linreg_R + self.nonlinearity(self.L2RgateFromL(x_L)) * (self.L2R1(x_L))
        elif self.bridge == 'scalargatesymFromTarget':
            pred_L = (1-self.nonlinearity(self.R2LgateFromL(x_L))) * linreg_L + self.nonlinearity(self.R2LgateFromL(x_L)) * (self.R2L1(x_R))
            pred_R = (1 - self.nonlinearity(self.L2RgateFromR(x_R))) * linreg_R + self.nonlinearity(self.L2RgateFromR(x_R)) * (self.L2R1(x_L))
        elif self.bridge == 'scalargateadjsym':
            pred_L = (1-self.nonlinearity(self.R2Lgate(x))) * (linreg_L + self.L2LCD(x_L)) + self.nonlinearity(self.R2Lgate(x)) * (self.R2L1(x_R))
            pred_R = (1 - self.nonlinearity(self.L2Rgate(x))) * (linreg_R + self.R2RCD(x_R)) + self.nonlinearity(self.L2Rgate(x)) * (self.L2R1(x_L))
        elif self.bridge == 'scalargateFull':
            pred_L = self.L2LFull(x_L) + self.nonlinearity(self.R2Lgate(x)) * self.R2LFull(x_R)
            pred_R = self.R2RFull(x_R) + self.nonlinearity(self.L2Rgate(x)) * self.L2RFull(x_L)
        elif self.bridge == 'scalargateFullpossig':
            pred_L = self.L2LFull(x_L) + (self.nonlinearity(self.R2Lgate(x))) * self.R2LFull(x_R)
            pred_R = self.R2RFull(x_R) + (self.nonlinearity(self.L2Rgate(x))) * self.L2RFull(x_L)
        elif self.bridge == 'scalargateFullsym':
            pred_L = (1 - self.nonlinearity(self.R2Lgate(x))) * self.L2LFull(x_L) + self.nonlinearity(self.R2Lgate(x)) * self.R2LFull(x_R)
            pred_R = (1 - self.nonlinearity(self.L2Rgate(x))) * self.R2RFull(x_R) + self.nonlinearity(self.L2Rgate(x)) * self.L2RFull(x_L)
        elif self.bridge == 'scalargateFullsympossig':
            pred_L = (1 - (self.nonlinearity(self.R2Lgate(x)))) * self.L2LFull(x_L) + (self.nonlinearity(self.R2Lgate(x))) * self.R2LFull(x_R)
            pred_R = (1 - (self.nonlinearity(self.L2Rgate(x)))) * self.R2RFull(x_R) + (self.nonlinearity(self.L2Rgate(x))) * self.L2RFull(x_L)

        elif self.bridge == 'interp':
            pred_L = linreg_L + self.R2Lgate(x) * (linreg_L - linregNB_L)
            pred_R = linreg_R + self.L2Rgate(x) * (linreg_R - linregNB_R)
        elif self.bridge == 'interpFromL':
            pred_L = linreg_L + self.R2LgateFromL(x_L) * (linreg_L - linregNB_L)
            pred_R = linreg_R + self.L2RgateFromL(x_L) * (linreg_R - linregNB_R)
        elif self.bridge == 'interpFromR':
            pred_L = linreg_L + self.R2LgateFromR(x_R) * (linreg_L - linregNB_L)
            pred_R = linreg_R + self.L2RgateFromR(x_R) * (linreg_R - linregNB_R)
        elif self.bridge == 'interpFromSource':
            pred_L = linreg_L + self.R2LgateFromR(x_R) * (linreg_L - linregNB_L)
            pred_R = linreg_R + self.L2RgateFromL(x_L) * (linreg_R - linregNB_R)
        elif self.bridge == 'interpFromTarget':
            pred_L = linreg_L + self.R2LgateFromL(x_L) * (linreg_L - linregNB_L)
            pred_R = linreg_R + self.L2RgateFromR(x_R) * (linreg_R - linregNB_R)
        elif self.bridge == 'scalargateCDsym':
            for p in self.L2L.parameters():
                p.requires_grad=False
            for p in self.R2R.parameters():
                p.requires_grad=False
            CD_proj_L = self.L2LCD(x_L)
            CD_proj_R = self.R2RCD(x_R)
            pred_L = (1-self.nonlinearity(self.R2Lgate(x))) * linreg_L + self.nonlinearity(self.R2Lgate(x)) * (self.R2L1(CD_proj_R))
            pred_R = (1 - self.nonlinearity(self.L2Rgate(x))) * linreg_R  + self.nonlinearity(self.L2Rgate(x)) * (self.L2R1(CD_proj_L))
        elif self.bridge == 'scalargatefxsym':
            pred_L = (1-self.nonlinearity(self.R2Lgate(x))) * linreg_L + self.nonlinearity(self.R2Lgate(x)) * linreg_R
            pred_R = (1 - self.nonlinearity(self.L2Rgate(x))) * linreg_R + self.nonlinearity(self.L2Rgate(x)) * linreg_L
        elif self.bridge == 'scalargatemixsym':
            pred_L = (1-self.nonlinearity(self.R2Lgate(x))) * (linreg_L + self.R2Lmix(x_R)) + self.nonlinearity(self.R2Lgate(x)) * (self.R2L1(x_R))
            pred_R = (1 - self.nonlinearity(self.L2Rgate(x))) * (linreg_R + self.L2Rmix(x_L)) + self.nonlinearity(self.L2Rgate(x)) * (self.L2R1(x_L))
        elif self.bridge == 'scalargatemult':
            pred_L = linreg_L + self.nonlinearity(self.R2LgatefromL(x_L)) * self.nonlinearity(self.R2LgatefromR(x_R)) * (self.R2L1(x_R))
            pred_R = linreg_R + self.nonlinearity(self.L2RgatefromL(x_L)) * self.nonlinearity(self.L2RgatefromR(x_R)) * (self.L2R1(x_L))
        elif self.bridge == 'scalargatemultsym':
            pred_L = (1 - self.nonlinearity(self.R2LgatefromL(x_L)) * self.nonlinearity(self.R2LgatefromR(x_R))) * linreg_L + self.nonlinearity(self.R2LgatefromL(x_L)) * self.nonlinearity(self.R2LgatefromR(x_R)) * (self.R2L1(x_R))
            pred_R = (1 - self.nonlinearity(self.L2RgatefromL(x_L)) * self.nonlinearity(self.L2RgatefromR(x_R))) * linreg_R + self.nonlinearity(self.L2RgatefromL(x_L)) * self.nonlinearity(self.L2RgatefromR(x_R)) * (self.L2R1(x_L))


        #return pred_L, pred_R, (self.nonlinearity(self.L2RgateFromR(x_R))), (self.nonlinearity(self.R2LgateFromL(x_L)))#(pred_L - self.L2LFull(x_L)), (pred_R - self.R2RFull(x_R))

        return pred_L, pred_R


