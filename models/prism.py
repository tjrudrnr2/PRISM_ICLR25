from collections import OrderedDict
from math import log
import math
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torch.nn.utils import spectral_norm
from utils.random import Bern

channel_multiplier = 1

class GetSubnet(autograd.Function):
    @staticmethod
    def forward(ctx, scores, bias_scores, k, scores_prune_threshold=-np.inf, bias_scores_prune_threshold=-np.inf):
        # Get the subnetwork by sorting the scores and using the top k%
        out = scores.clone()
        _, idx = scores.flatten().sort()
        j = int((1 - k) * scores.numel())
        
        # flat_out and out access the same memory.
        flat_out = out.flatten()
        flat_out[idx[:j]] = False
        flat_out[idx[j:]] = True
        
        bias_out = bias_scores.clone()
        _, idx = bias_scores.flatten().sort()
        j = int((1 - k) * bias_scores.numel())

        bias_flat_out = bias_out.flatten()
        bias_flat_out[idx[:j]] = 0
        bias_flat_out[idx[j:]] = 1
        
        # s_subnet = torch.sigmoid(scores)           
        # s_bias_subnet = torch.sigmoid(bias_scores)
        # out = torch.where(s_subnet >= 0.7, 1, 0)
        # bias_out = torch.where(s_bias_subnet >= 0.5, 1, 0)
        
        # out = torch.gt(scores, torch.ones_like(scores)*scores_prune_threshold).float()
        # bias_out = torch.gt(bias_scores, torch.ones_like(bias_scores)*bias_scores_prune_threshold).float()            
        # out = Bern.apply(out)
        # bias_out = Bern.apply(bias_out)            
        
        return out, bias_out

    @staticmethod
    def backward(ctx, g_1, g_2):
        # send the gradient g straight-through on the backward pass.
        return g_1, g_2, None, None, None


class SubnetLinear(nn.Linear):
    def __init__(self, opt, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.flag = nn.Parameter(torch.ones(self.weight.size()))
        self.scores = nn.Parameter(torch.Tensor(self.weight.size()))
        self.opt = opt
        
        if opt.bias:
            if self.bias != None:
                self.bias_flag = nn.Parameter(torch.ones(self.bias.size()))
                self.bias_scores = nn.Parameter(torch.Tensor(self.bias.size()))
            else:
                self.bias_flag = nn.Parameter(torch.tensor(0.9))
                self.bias_scores = nn.Parameter(torch.tensor(0.9))    
        else:
            self.bias_flag = nn.Parameter(torch.tensor(0.9))
            self.bias_scores = nn.Parameter(torch.tensor(0.9))

        self.scores_prune_threshold = -np.inf
        self.bias_scores_prune_threshold = -np.inf

        if opt.score_init == "dense":
            nn.init.uniform_(self.scores, a=-0.1, b=0.1)
        else:
            nn.init.kaiming_uniform_(self.scores, a=math.sqrt(5))
        nn.init.uniform_(self.bias_scores, a=-1.0, b=1.0)

        self.flag.requires_grad_(False)
        self.bias_flag.requires_grad_(False)
        
        self.weight.requires_grad_(False)
        if opt.bias:
            if self.bias != None:
                self.bias.requires_grad_(False)
            else:
                pass
        
        #weight initialization
        #signed kaiming constant
        if opt.init == "Constant":
            fan = nn.init._calculate_correct_fan(self.weight, "fan_in")
            gain = nn.init.calculate_gain("relu")
            std = gain/math.sqrt(fan)
            self.weight.data = self.weight.data.sign()*std
        elif opt.init == "G_Normal":
        ##gaussian normal
            nn.init.normal_(self.weight, std = 10.0)
        ##kaiming normal
        elif opt.init =="K_Normal":
            if opt.scale_fan == True:
                fan = nn.init._calculate_correct_fan(self.weight, "fan_in")
                # fan = fan * opt.sparsity
                gain = nn.init.calculate_gain("relu")
                std = gain / math.sqrt(fan)
                with torch.no_grad():
                    self.weight.data.normal_(0, std)
            else:
                nn.init.kaiming_normal_(self.weight, mode = "fan_in", nonlinearity = "relu")
                
    def set_prune_rate(self, prune_rate):
        self.prune_rate = prune_rate

    @property
    def clamped_scores(self):
        return self.scores.abs()

    def forward(self, x):
        # print("cur key mask mechanism is top k")
        # subnet, bias_subnet = GetSubnet.apply(self.scores.abs(), self.bias_scores.abs(), 0.5)
        
        # print("cur key mask mechanism is random")
        # subnet = torch.bernoulli(torch.empty(self.scores.shape).uniform_(0, 1)).to(f'cuda:{self.opt.gpunum}')
        # bias_subnet = torch.bernoulli(torch.empty(self.scores.shape).uniform_(0, 1)).to(f'cuda:{self.opt.gpunum}')
        
        if self.opt.sparsity > 0:
            # print("cur key mask mechanism is top k")
            subnet, bias_subnet = GetSubnet.apply(self.scores.abs(), self.bias_scores.abs(), self.opt.sparsity)
        elif self.opt.sparsity == 0:
            s_subnet = torch.sigmoid(self.scores)           
            s_bias_subnet = torch.sigmoid(self.bias_scores)
            subnet = Bern.apply(s_subnet)
            bias_subnet = Bern.apply(s_bias_subnet)
            
        # subnet = torch.where(s_subnet >= 0.5, 1, 0)
        # # subnet = torch.where(subnet < 0.5, 0, 1)
        # bias_subnet = torch.where(s_bias_subnet >= 0.5, 1, 0)
        # # bias_subnet = torch.where(bias_subnet < 0.5, 0, 1)
        
        # print("cur key mask mechanism is random")
        # s_subnet = torch.sigmoid(self.scores)           
        # s_bias_subnet = torch.sigmoid(self.bias_scores)
        # subnet = Bern.apply(s_subnet)
        # bias_subnet = Bern.apply(s_bias_subnet)
        
        w = self.weight * subnet
        
        if self.opt.bias:
            if self.bias != None:
                b = self.bias * bias_subnet
            else:
                b = self.bias
        else:
            b = self.bias
    

        return F.linear(x, w, b)


class SubnetConv(nn.Conv2d):
    def __init__(self, opt, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.flag = nn.Parameter(torch.ones(self.weight.size()))
        self.scores = nn.Parameter(torch.Tensor(self.weight.size()))

        self.opt = opt

        if opt.bias:
            if self.bias != None:
                self.bias_flag = nn.Parameter(torch.ones(self.bias.size()))
                self.bias_scores = nn.Parameter(torch.Tensor(self.bias.size()))
            else:
                self.bias_flag = nn.Parameter(torch.tensor(0.9))
                self.bias_scores = nn.Parameter(torch.tensor(0.9))
        else:
            self.bias_flag = nn.Parameter(torch.tensor(0.9))
            self.bias_scores = nn.Parameter(torch.tensor(0.9))

        self.scores_prune_threshold = -np.inf
        self.bias_scores_prune_threshold = -np.inf
        
        self.c = np.e * np.sqrt(1/(self.kernel_size[0]**2 * self.in_channels))
        
        self.flag.requires_grad_(False)
        self.bias_flag.requires_grad_(False)
        
        self.weight.requires_grad_(False)
        if opt.bias:
            if self.bias != None:
                self.bias.requires_grad_(False)
            else:
                pass
        
        #weight initialization
        #signed kaiming constant
        if opt.init == "Constant":
            fan = nn.init._calculate_correct_fan(self.weight, "fan_in")
            gain = nn.init.calculate_gain("relu")
            std = gain/math.sqrt(fan)
            self.weight.data = self.weight.data.sign()*std
        ##gaussian normal
        elif opt.init == "G_Normal":
            nn.init.normal_(self.weight, std = 10.0)
        ##kaiming normal
        elif opt.init =="K_Normal":
            if opt.scale_fan == True:
                fan = nn.init._calculate_correct_fan(self.weight, "fan_in")
                # fan = fan * opt.sparsity
                gain = nn.init.calculate_gain("relu")
                std = gain / math.sqrt(fan)
                with torch.no_grad():
                    self.weight.data.normal_(0, std)
            else:
                nn.init.kaiming_normal_(self.weight, mode = "fan_in", nonlinearity = "relu")
        elif opt.init == "ME_init":
            self.c = np.e * np.sqrt(1/(self.kernel_size[0]**2 * self.in_channels))
            arr_weights = np.random.choice([-self.c, self.c], size=(self.out_channels, self.in_channels, self.kernel_size[0], self.kernel_size[0]))
            self.weight = nn.Parameter(torch.tensor(arr_weights, requires_grad=False, dtype=torch.float))
        
        # Score initialization
        if opt.score_init == "dense":
            nn.init.uniform_(self.scores, a=-0.1, b=0.1)
        elif opt.score_init == "ME_init":
            self.scores = nn.Parameter(torch.randn_like(self.weight, requires_grad=True))
        else:
            nn.init.kaiming_uniform_(self.scores, a=math.sqrt(5))
        nn.init.uniform_(self.bias_scores, a=-1.0, b=1.0)
        
    def set_prune_rate(self, prune_rate):
        self.prune_rate = prune_rate

    @property
    def clamped_scores(self):
        return self.scores.abs()


    def forward(self, x):
        # print("cur key mask mechanism is top k")
        # subnet, bias_subnet = GetSubnet.apply(self.scores.abs(), self.bias_scores.abs(), 0.5)
        
        # print("cur key mask mechanism is random")
        # subnet = torch.bernoulli(torch.empty(self.scores.shape).uniform_(0, 1)).to(f'cuda:{self.opt.gpunum}')
        # bias_subnet = torch.bernoulli(torch.empty(self.scores.shape).uniform_(0, 1)).to(f'cuda:{self.opt.gpunum}')
        
        # print("cur key mask mechanism is thresholding")
        # s_subnet = torch.sigmoid(self.scores)           
        # s_bias_subnet = torch.sigmoid(self.bias_scores)
        # subnet = torch.where(s_subnet >= 0.5, 1, 0)
        # # subnet = torch.where(subnet < 0.5, 0, 1)
        # bias_subnet = torch.where(s_bias_subnet >= 0.5, 1, 0)
        # # bias_subnet = torch.where(bias_subnet < 0.5, 0, 1)
                
        if self.opt.sparsity > 0:
            # print("cur key mask mechanism is top k")
            subnet, bias_subnet = GetSubnet.apply(self.scores.abs(), self.bias_scores.abs(), self.opt.sparsity)
        elif self.opt.sparsity == 0:
            # print("cur key mask mechanism is thresholding")
            s_subnet = torch.sigmoid(self.scores)           
            s_bias_subnet = torch.sigmoid(self.bias_scores)
            subnet = Bern.apply(s_subnet)
            bias_subnet = Bern.apply(s_bias_subnet)
        
        w = self.weight * subnet
        
        if self.opt.bias:
            if self.bias != None:
                s_bias_subnet = torch.sigmoid(self.bias_scores)
                bias_subnet = Bern.apply(s_bias_subnet)            
                b = self.bias * bias_subnet
            else:
                b = self.bias
        else:
            b = self.bias

        x = F.conv2d(
            x, w, b, self.stride, self.padding, self.dilation, self.groups
        )
        return x
        
        # return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class SubnetConvT(nn.ConvTranspose2d):
    def __init__(self, opt, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.flag = nn.Parameter(torch.ones(self.weight.size()))
        self.scores = nn.Parameter(torch.Tensor(self.weight.size()))
        
        self.opt = opt

        if opt.bias:
            if self.bias != None:
                self.bias_flag = nn.Parameter(torch.ones(self.bias.size()))
                self.bias_scores = nn.Parameter(torch.Tensor(self.bias.size()))
            else:
                self.bias_flag = nn.Parameter(torch.tensor(0.9))
                self.bias_scores = nn.Parameter(torch.tensor(0.9))    
        else:
            self.bias_flag = nn.Parameter(torch.tensor(0.9))
            self.bias_scores = nn.Parameter(torch.tensor(0.9))

        self.scores_prune_threshold = -np.inf
        self.bias_scores_prune_threshold = -np.inf
        
        self.c = np.e * np.sqrt(1/(self.kernel_size[0]**2 * self.in_channels))

        self.flag.requires_grad_(False)
        self.bias_flag.requires_grad_(False)
        
        self.weight.requires_grad_(False)
        if opt.bias:
            if self.bias != None:
                self.bias.requires_grad_(False)
            else:
                pass
        
        #weight initialization
        #signed kaiming constant
        if opt.init == "Constant":
            fan = nn.init._calculate_correct_fan(self.weight, "fan_in")
            gain = nn.init.calculate_gain("relu")
            std = gain/math.sqrt(fan)
            self.weight.data = self.weight.data.sign()*std
        elif opt.init == "G_Normal":
        ##gaussian normal
            nn.init.normal_(self.weight, std = 10.0)
        ##kaiming normal
        elif opt.init =="K_Normal":
            if opt.scale_fan == True:
                fan = nn.init._calculate_correct_fan(self.weight, "fan_in")
                # fan = fan * opt.sparsity
                gain = nn.init.calculate_gain("relu")
                std = gain / math.sqrt(fan)
                with torch.no_grad():
                    self.weight.data.normal_(0, std)
            else:
                nn.init.kaiming_normal_(self.weight, mode = "fan_in", nonlinearity = "relu")         
        elif opt.init == "ME_init":
            self.c = np.e * np.sqrt(1/(self.kernel_size[0]**2 * self.in_channels))
            arr_weights = np.random.choice([-self.c, self.c], size=(self.in_channels, self.out_channels, self.kernel_size[0], self.kernel_size[0]))
            self.weight = nn.Parameter(torch.tensor(arr_weights, requires_grad=False, dtype=torch.float))
        
        # Score initialization
        if opt.score_init == "dense":
            nn.init.uniform_(self.scores, a=-0.1, b=0.1)
        elif opt.score_init == "ME_init":
            self.scores = nn.Parameter(torch.randn_like(self.weight, requires_grad=True))
        else:
            nn.init.kaiming_uniform_(self.scores, a=math.sqrt(5))
        nn.init.uniform_(self.bias_scores, a=-1.0, b=1.0)

    def set_prune_rate(self, prune_rate):
        self.prune_rate = prune_rate

    @property
    def clamped_scores(self):
        return self.scores.abs()

    def forward(self, x):
        # print("cur key mask mechanism is top k")
        # subnet, bias_subnet = GetSubnet.apply(self.scores.abs(), self.bias_scores.abs(), 0.5)
        
        # print("cur key mask mechanism is random")
        # subnet = torch.bernoulli(torch.empty(self.scores.shape).uniform_(0, 1)).to(f'cuda:{self.opt.gpunum}')
        # bias_subnet = torch.bernoulli(torch.empty(self.scores.shape).uniform_(0, 1)).to(f'cuda:{self.opt.gpunum}')
        
        # print("cur key mask mechanism is thresholding")
        # subnet, bias_subnet = GetSubnet.apply(self.scores.abs(), self.bias_scores.abs(), 0.5)
                
        # s_subnet = torch.sigmoid(self.scores)           
        # s_bias_subnet = torch.sigmoid(self.bias_scores)
        # subnet = Bern.apply(s_subnet)
        # bias_subnet = Bern.apply(s_bias_subnet)
        
        if self.opt.sparsity > 0:
            # print("cur key mask mechanism is top k")
            subnet, bias_subnet = GetSubnet.apply(self.scores.abs(), self.bias_scores.abs(), self.opt.sparsity)
        elif self.opt.sparsity == 0:
            s_subnet = torch.sigmoid(self.scores)           
            s_bias_subnet = torch.sigmoid(self.bias_scores)
            subnet = Bern.apply(s_subnet)
            bias_subnet = Bern.apply(s_bias_subnet)
        
        w = self.weight * subnet
        
        if self.opt.bias:
            if self.bias != None:
                b = self.bias * bias_subnet
            else:
                b = self.bias
        else:
            b = self.bias
            
        x = F.conv_transpose2d(
            x, w, b, self.stride, self.padding, self.output_padding, self.groups, self.dilation
        )
            
        return x
        
        # return F.conv_transpose2d(
        #     x, self.weight, self.bias, self.stride, self.padding, self.output_padding, self.groups, self.dilation
        # )


class ResnetG(nn.Module):
    def __init__(self, opt, nz, nc, ndf, imageSize = 32, adaptFilterSize = False, useConvAtSkipConn = False):
        super(ResnetG, self).__init__()
        self.nz = nz
        self.ndf = ndf
        
        self.opt = opt

        if adaptFilterSize == True and useConvAtSkipConn == False:
            useConvAtSkipConn = True
        
        numUpsampleBlocks = int(log(imageSize, 2)) - 2 
        
        numLayers = numUpsampleBlocks + 1
        filterSizePerLayer = [ndf] * numLayers
        if adaptFilterSize:
            for i in range(numLayers - 1, -1, -1):
                if i == numLayers - 1:
                    filterSizePerLayer[i] = ndf
                else:
                    filterSizePerLayer[i] = filterSizePerLayer[i+1]*2
            
        firstL = SubnetConvT(self.opt, nz, int(filterSizePerLayer[0] * channel_multiplier), 4, 1, 0, bias=False)
        lastL  = SubnetConv(self.opt, int(filterSizePerLayer[-1] * channel_multiplier), nc, 3, stride=1, padding=1, bias=self.opt.bias)

        nnLayers = OrderedDict()
        # first deconv goes from the z size
        nnLayers["firstConv"]   = firstL
        
        layerNumber = 1
        for i in range(numUpsampleBlocks):
            nnLayers["resblock_%d"%i] = ResidualBlockG(self.opt, filterSizePerLayer[layerNumber-1], filterSizePerLayer[layerNumber], stride=2, useConvAtSkipConn = useConvAtSkipConn)
            layerNumber += 1
        nnLayers["batchNorm"] = nn.BatchNorm2d(int(filterSizePerLayer[-1] * channel_multiplier), affine = opt.affine)
        nnLayers["relu"]      = nn.ReLU()
        nnLayers["lastConv"]  = lastL
        nnLayers["tanh"]      = nn.Tanh()

        self.net = nn.Sequential(nnLayers)

    def forward(self, input):
        return self.net(input)

# class ResnetG(nn.Module):
#     def __init__(self, args, nz, nc, ndf, imageSize = 32, adaptFilterSize = False, useConvAtSkipConn = False):
#         super(ResnetG, self).__init__()
#         self.nz = nz
#         self.ndf = ndf

#         if adaptFilterSize == True and useConvAtSkipConn == False:
#             useConvAtSkipConn = True
#         #     logger.warn("WARNING: In ResnetG, setting useConvAtSkipConn to True because adaptFilterSize is True.")

#         numUpsampleBlocks = int(log(imageSize, 2)) - 2 
        
#         numLayers = numUpsampleBlocks + 1
#         filterSizePerLayer = [ndf] * numLayers
#         if adaptFilterSize:
#             for i in range(numLayers - 1, -1, -1):
#                 if i == numLayers - 1:
#                     filterSizePerLayer[i] = ndf
#                 else:
#                     filterSizePerLayer[i] = filterSizePerLayer[i+1]*2
            
#         firstL = nn.ConvTranspose2d(nz, filterSizePerLayer[0], 4, 1, 0, bias=False)
#         nn.init.xavier_uniform(firstL.weight.data, 1.)
#         lastL  = nn.Conv2d(filterSizePerLayer[-1], nc, 3, stride=1, padding=1)
#         nn.init.xavier_uniform(lastL.weight.data, 1.)

#         nnLayers = OrderedDict()
#         # first deconv goes from the z size
#         nnLayers["firstConv"]   = firstL
        
#         layerNumber = 1
#         for i in range(numUpsampleBlocks):
#             nnLayers["resblock_%d"%i] = ResidualBlockG(args, filterSizePerLayer[layerNumber-1], filterSizePerLayer[layerNumber], stride=2, useConvAtSkipConn = useConvAtSkipConn)
#             layerNumber += 1
#         nnLayers["batchNorm"] = nn.BatchNorm2d(filterSizePerLayer[-1])
#         nnLayers["relu"]      = nn.ReLU()
#         nnLayers["lastConv"]  = lastL
#         nnLayers["tanh"]      = nn.Tanh()

#         self.net = nn.Sequential(nnLayers)

#     def forward(self, input):
#         print("current is GFMN and unfreeze weight")
        
#         return self.net(input)

class Upsample(nn.Module):
    def __init__(self, scale_factor=2, size=None):
        super(Upsample, self).__init__()
        self.upsample = F.upsample_nearest
        self.size = size
        self.scale_factor = scale_factor
        
    def forward(self, x):
        x = self.upsample(x, size=self.size, scale_factor = self.scale_factor)
        return x

# class ResidualBlockG(nn.Module):

#     def __init__(self, opt, in_channels, out_channels, stride=1, useConvAtSkipConn = False):
#         super(ResidualBlockG, self).__init__()

#         self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1)
#         self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1)
        
#         if useConvAtSkipConn:
#             self.conv_bypass = nn.Conv2d(in_channels, out_channels, 1, 1, padding=0)
#             nn.init.xavier_uniform(self.conv_bypass.weight.data, 1.)
        
#         nn.init.xavier_uniform(self.conv1.weight.data, 1.)
#         nn.init.xavier_uniform(self.conv2.weight.data, 1.)

#         self.model = nn.Sequential(
#             nn.BatchNorm2d(in_channels),
#             nn.ReLU(),
#             Upsample(scale_factor=2),
#             self.conv1,
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(),
#             self.conv2
#             )
#         self.bypass = nn.Sequential()
#         if stride != 1:
#             if useConvAtSkipConn:
#                 self.bypass = nn.Sequential(self.conv_bypass, Upsample(scale_factor=2))
#             else:
#                 self.bypass = Upsample(scale_factor=2)

#     def forward(self, x):
#         # A, B = self.model(x), self.bypass(x)
#         return self.model(x) + self.bypass(x)

class ResidualBlockG(nn.Module):

    def __init__(self, opt, in_channels, out_channels, stride=1, useConvAtSkipConn = False):
        super(ResidualBlockG, self).__init__()
        
        self.opt = opt

        self.conv1 = SubnetConv(self.opt, int(in_channels * channel_multiplier), int(out_channels * channel_multiplier), 3, 1, padding=1, bias=opt.bias)
        self.conv2 = SubnetConv(self.opt, int(out_channels * channel_multiplier), int(out_channels * channel_multiplier), 3, 1, padding=1, bias=opt.bias)
        
        if useConvAtSkipConn:
            self.conv_bypass = SubnetConv(self.opt, int(in_channels * channel_multiplier), int(out_channels * channel_multiplier), 1, 1, padding=0, bias=opt.bias)
#            nn.init.xavier_uniform(self.conv_bypass.weight.data, 1.)
        
#        nn.init.xavier_uniform(self.conv1.weight.data, 1.)
#        nn.init.xavier_uniform(self.conv2.weight.data, 1.)

        self.model = nn.Sequential(
            nn.BatchNorm2d(int(in_channels * channel_multiplier), affine = opt.affine),
            nn.ReLU(),
            Upsample(scale_factor=2),
            self.conv1,
            nn.BatchNorm2d(int(out_channels * channel_multiplier), affine = opt.affine),
            nn.ReLU(),
            self.conv2
            )
        self.bypass = nn.Sequential()
        if stride != 1:
            if useConvAtSkipConn:
                self.bypass = nn.Sequential(self.conv_bypass, Upsample(scale_factor=2))
            else:
                self.bypass = Upsample(scale_factor=2)

    def forward(self, x):
        return self.model(x) + self.bypass(x)