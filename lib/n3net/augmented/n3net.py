'''
Author: Tobias Plötz, TU Darmstadt (tobias.ploetz@visinf.tu-darmstadt.de)

This file is part of the implementation as described in the NIPS 2018 paper:
Tobias Plötz and Stefan Roth, Neural Nearest Neighbors Networks.
Please see the file LICENSE.txt for the license governing this code.
'''

import math
import torch as th
import torch.nn as nn
from . import nl_agg

# class Conv2dFlops(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
#                  padding=0,bais=True):
#         self.super().__init__()
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.kernel_size = kernel_size
#         self.stride = stride
#         self.padding = padding
#         self.bias = bais
#         self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
#                               stride=stride, padding=padding, bias=bias)
#     def forward(self,img):
#         return self.conv(img)

#     def flops(self, H, W):
#         flops = H*W*self.in_channels*self.out_channels*(self.kernel_size**2+self.bias)
#         return flops

# class BatchNormFlops(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         pass
    # def
    # flops += H*W*self.out_channel

def convnxn(in_planes, out_planes, kernelsize, stride=1, bias=False):
    padding = kernelsize//2
    # return Conv2dFlops(in_planes, out_planes, kernel_size=kernelsize, stride=stride, padding=padding, bias=bias)
    return nn.Conv2d(in_planes, out_planes, kernel_size=kernelsize, stride=stride, padding=padding, bias=bias)

def dncnn_batchnorm_init(m, kernelsize=3, b_min=0.025):
    r"""
    Reproduces batchnorm initialization from DnCNN
    https://github.com/cszn/DnCNN/blob/master/TrainingCodes/DnCNN_TrainingCodes_v1.1/DnCNN_init_model_64_25_Res_Bnorm_Adam.m
    """
    n = kernelsize**2 * m.num_features
    m.weight.data.normal_(0, math.sqrt(2. / (n)))
    m.weight.data[(m.weight.data > 0) & (m.weight.data <= b_min)] = b_min
    m.weight.data[(m.weight.data < 0) & (m.weight.data >= -b_min)] = -b_min
    m.weight.data = m.weight.data.abs()
    m.bias.data.zero_()
    m.momentum = 0.001


class DnCNN(nn.Module):
    r"""
    Implements a DnCNN network
    """
    def __init__(self,  nplanes_in, nplanes_out, features, kernel, depth, residual, bn):
        r"""
        :param nplanes_in: number of of input feature channels
        :param nplanes_out: number of of output feature channels
        :param features: number of of hidden layer feature channels
        :param kernel: kernel size of convolution layers
        :param depth: number of convolution layers (minimum 2)
        :param residual: whether to add a residual connection from input to output
        :param bn:  whether to add batchnorm layers
        """
        super(DnCNN, self).__init__()

        self.residual = residual
        self.nplanes_out = nplanes_out
        self.nplanes_in = nplanes_in
        self.kernelsize = kernel
        self.nplanes_residual = None

        print(nplanes_in, features, kernel)
        self.conv1 = convnxn(nplanes_in, features, kernelsize=kernel, bias=True)
        self.bn1 = nn.BatchNorm2d(features) if bn else nn.Sequential()
        self.relu = nn.ReLU(inplace=True)
        layers = []
        for i in range(depth-2):
            layers += [convnxn(features, features, kernel),
                       nn.BatchNorm2d(features)  if bn else nn.Sequential(),
                       self.relu]
        self.layer1 = nn.Sequential(*layers)
        self.conv2 = convnxn(features , nplanes_out, kernelsize=kernel, bias=True)

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / (n)))
            elif isinstance(m, nn.BatchNorm2d):
                dncnn_batchnorm_init(m, kernelsize=self.kernelsize, b_min=0.025)

    def forward(self, x):
        shortcut = x

        x = self.conv1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.conv2(x)

        nplanes_residual = self.nplanes_residual or self.nplanes_in
        if self.residual:
            nshortcut = min(self.nplanes_in, self.nplanes_out, nplanes_residual)
            x[:,:nshortcut,:,:] = x[:,:nshortcut,:,:] + shortcut[:,:nshortcut,:,:]

        return x


def cnn_from_def(cnn_opt):
    kernel = cnn_opt.get("kernel",3)
    padding = (kernel-1)//2
    cnn_bn = cnn_opt.get("bn",True)
    cnn_depth = cnn_opt.get("depth",0)
    cnn_channels = cnn_opt.get("features")
    cnn_outchannels = cnn_opt.get("nplanes_out",)
    chan_in = cnn_opt.get("nplanes_in")

    if cnn_depth == 0:
        cnn_outchannels=chan_in

    cnn_layers = []
    relu = nn.ReLU(inplace=True)

    for i in range(cnn_depth-1):
        cnn_layers.extend([
            nn.Conv2d(chan_in,cnn_channels,kernel, 1, padding, bias=not cnn_bn),
            nn.BatchNorm2d(cnn_channels) if cnn_bn else nn.Sequential(),
            relu
        ])
        chan_in = cnn_channels

    if cnn_depth > 0:
        cnn_layers.append(
            nn.Conv2d(chan_in,cnn_outchannels,kernel, 1, padding, bias=True)
        )

    net = nn.Sequential(*cnn_layers)
    net.nplanes_out = cnn_outchannels
    net.nplanes_in = cnn_opt.get("nplanes_in")
    return net


class N3Block(nn.Module):
    r"""
    N3Block operating on a 2D images
    """
    def __init__(self, nplanes_in,
                 k=7, patchsize=10, stride=1, dilation=1,
                 ws=29, wt=0, pt=1, batch_size=None,
                 use_cts_topk = False, temp_opt={}, embedcnn_opt={}):
        r"""
        :param nplanes_in: number of input features
        :param k: number of neighbors to sample
        :param patchsize: size of patches that are matched
        :param stride: stride with which patches are extracted
        :param nl_match_window: size of matching window around each patch,
            i.e. the nl_match_window x nl_match_window patches around a query patch
            are used for matching
        :param temp_opt: options for handling the the temperature parameter
        :param embedcnn_opt: options for the embedding cnn, also shared by temperature cnn
        """
        super(N3Block, self).__init__()

        # -- init --
        self.k = k
        self.ps = patchsize
        self.patchsize = patchsize
        self.pt = pt
        self.stride = stride
        self.dilation = dilation
        self.batch_size = batch_size
        self.ws,self.wt = ws,wt
        self.use_cts_topk = use_cts_topk

        # -- patch embedding --
        embedcnn_opt["nplanes_in"] = nplanes_in
        self.embedcnn = cnn_from_def(embedcnn_opt)

        # -- temperature cnn --
        with_temp = temp_opt.get("external_temp")
        if with_temp:
            tempcnn_opt = dict(**embedcnn_opt)
            tempcnn_opt["nplanes_out"] = 1
            self.tempcnn = cnn_from_def(tempcnn_opt)
        else:
            self.tempcnn = None

        # -- n3agg --
        self.nplanes_in = nplanes_in
        # self.nplanes_out = (k+1) * nplanes_in
        k_shape = 7 if self.use_cts_topk else 1
        self.nplanes_out = (k_shape+1) * nplanes_in # fixed "k == 7"
        self.n3aggregation = nl_agg.N3Aggregation2D(
            k=k, patchsize=patchsize, stride=stride, dilation=dilation,
            ws=ws, wt=wt, pt=pt, batch_size=batch_size,
            use_cts_topk=use_cts_topk, temp_opt=temp_opt)

        self.reset_parameters()

    def forward(self, x, flows):
        if self.k <= 0:
            return x

        xe = self.embedcnn(x)
        ye = xe
        xg = x
        if self.tempcnn is not None:
            log_temp = self.tempcnn(x)
        else:
            log_temp = None

        x = self.n3aggregation(xg,xe,ye,None,log_temp,flows)
        return x

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm2d)):
                dncnn_batchnorm_init(m, kernelsize=3, b_min=0.025)

class N3Net(nn.Module):
    r"""
    A N3Net interleaves DnCNNS for local processing with N3Blocks for non-local processing
    """
    def __init__(self, nplanes_in, nplanes_out, nplanes_interm, nblocks,
                 residual, block_opt, nl_temp_opt, embedcnn_opt,
                 ws=29, wt=0, k=7, stride=5, dilation=1,
                 patchsize=10, pt=1, batch_size=None, use_cts_topk=False):

        r"""
        :param nplanes_in: number of input features
        :param nplanes_out: number of output features
        :param nplanes_interm: number of intermediate features, i.e. number of output features for the DnCNN sub-networks
        :param nblocks: number of DnCNN sub-networks
        :param block_opt: options passed to DnCNNs
        :param nl_opt: options passed to N3Blocks
        :param residual: whether to have a global skip connection
        """
        super(N3Net, self).__init__()
        self.nplanes_in = nplanes_in
        self.nplanes_out = nplanes_out
        self.nblocks = nblocks
        self.residual = residual
        # print(nplanes_in,nplanes_out,nplanes_interm)

        nin = nplanes_in
        cnns,nls = [],[]
        for i in range(nblocks-1):
            cnns.append(DnCNN(nin, nplanes_interm, **block_opt))
            nl = N3Block(nplanes_interm,k=k,patchsize=patchsize,
                         stride=stride,dilation=dilation,
                         ws=ws,wt=wt,pt=pt,batch_size=batch_size,
                         temp_opt=nl_temp_opt,
                         use_cts_topk=use_cts_topk,
                         embedcnn_opt=embedcnn_opt)
            nin = nl.nplanes_out
            nls.append(nl)

        nout = nplanes_out
        cnns.append(DnCNN(nin, nout, **block_opt))

        self.nls = nn.Sequential(*nls)
        self.blocks = nn.Sequential(*cnns)

    def forward(self, x, flows=None):
        # print("refactored.")
        shortcut = x
        for i in range(self.nblocks-1):
            x = self.blocks[i](x)
            x = self.nls[i](x, flows)

        x = self.blocks[-1](x)
        if self.residual:
            nshortcut = min(self.nplanes_in, self.nplanes_out)
            x[:,:nshortcut,:,:] = x[:,:nshortcut,:,:] + shortcut[:,:nshortcut,:,:]

        return x

