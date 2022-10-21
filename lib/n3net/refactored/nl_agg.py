
import math
from math import log

import numpy as np
import torch as th
from einops import rearrange,repeat

import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import dnls
from dnls.utils.pads import comp_pads
from n3net.utils.misc import get_flows
from .non_local import NeuralNearestNeighbors

class N3AggregationBase(nn.Module):
    r"""
    Domain agnostic base class for computing neural nearest neighbors
    """
    def __init__(self, k, temp_opt={}):
        r"""
        :param k: Number of neighbor volumes to compute
        :param temp_opt: options for handling temperatures, see `NeuralNearestNeighbors`
        """
        super(N3AggregationBase, self).__init__()
        self.k = k
        self.temp_opt = temp_opt
        self.nnn = NeuralNearestNeighbors(k, temp_opt=temp_opt)

    def forward(self, x, xe, ye, log_temp,
                search, unfold, fold, wpsum,
                qindex, nbatch_i):
        r"""
        :param x: database items, shape BxNxF
        :param xe: embedding of database items, shape BxNxE
        :param ye: embedding of query items, shape BxMxE
        :param y: query items, if None then y=x is assumed, shape BxMxF
        :param I: Indexing tensor defining O potential neighbors for each query item
            shape BxMxO
        :param log_temp: optional log temperature
        :return:
        """

        # -- compute distance --
        dists,inds = search(xe[None,:],qindex,nbatch_i,ye[None,:])
        # dists,inds = dists[0],inds[0]
        dists = -dists
        # print("dists.shape: ",dists.shape)
        # print(dists[0])
        # print(th.where(th.abs(dists[0] + 0.1306)<1e-3))
        # print(th.where(th.abs(dists[0] + 0.2401)<1e-3))
        # print(th.where(th.abs(dists[0] + 0.3601)<1e-3))
        # print(th.where(th.abs(dists[0] + 0.5393)<1e-3))
        # print(th.where(th.abs(dists[0] + 0.6493)<1e-3))
        # print("[ref] D: ",dists[0,:5])
        # print("[ref] D: ",dists[80,:5])
        # print("[ref] D.shape: ",dists.shape)
        # dists,inds = remove_qinds(dists,inds,qinds)
        # print(dists.shape)
        # print(inds.shape)

        # -- log_temp patches --
        ps = unfold.ps
        lt_patches = unfold(log_temp,qindex,nbatch_i)
        lt_patches = rearrange(lt_patches,'b 1 1 c h w -> b 1 (h w) c')
        if self.temp_opt["avgpool"]:
            lt_patches = lt_patches.mean(dim=2)
        else:
            lt_patches = lt_patches[:,:,lt_patches.shape[2]//2,:].contiguous()
        # print("lt_patches.shape: ",lt_patches.shape)

        # -- compute aggregation weights --
        dists = dists[None,:]
        W = self.nnn(dists, log_temp=lt_patches)[0]
        nheads,b,_ = W.shape

        # -- weighted patch sum --
        # z_patches = []
        # for ki in range(1):#self.k):
        #     W_ki = W[...,ki].contiguous()
        #     z_patches_ki = wpsum(x,W_ki,inds).view(b,-1)
        #     z_patches_ki = z_patches_ki.view(b,-1,ps,ps)
        #     # print("z_patches_ki.shape: ",z_patches_ki.shape)
        #     z_patches.append(z_patches_ki)
        # z_patches = th.stack(z_patches,1)

        c = x.shape[1]
        ps = wpsum.ps
        # z_patches = th.zeros((b,self.k,c,ps,ps),device=W.device,dtype=th.float32)
        print("x.shape ",x.shape)
        print("W.shape: ",W.shape)
        z_patches = wpsum(x[None,:],W[None,:],inds[None,:]) # b self.k c h w
        z_patches = z_patches[0]

        # z_patches = repeat(z_patches,'b 1 c h w -> b k c h w',k=self.k)
        # print("z_patches.shape: ",z_patches.shape)
        # print(z_patches[0,0,0,:3,:3])
        # print(z_patches[0,1,0,:7,:7])
        # print(z_patches[0,-1,0,:3,:3])
        # exit(0)

        # -- fold into video --
        ps = unfold.ps
        shape_str = 'b k c ph pw -> b 1 1 (k c) ph pw'
        z_patches = rearrange(z_patches,shape_str,ph=ps,pw=ps)
        # ones = th.ones_like(z_patches)
        fold(z_patches[None,:],qindex)
        # wfold(ones,qindex)

class N3Aggregation2D(nn.Module):
    r"""
    Computes neural nearest neighbors for image data based on extracting patches
    in strides.
    """
    def __init__(self, k=7, patchsize=10, stride=1, dilation=1,
                 ws=29, wt=0, pt=1, batch_size=None, temp_opt={}):
        r"""
        :param indexing: function for creating index tensor
        :param k: number of neighbor volumes
        :param patchsize: size of patches that are matched
        :param stride: stride with which patches are extracted
        :param temp_opt: options for handling temperatures, see `NeuralNearestNeighbors`
        """
        super(N3Aggregation2D, self).__init__()
        self.patchsize = patchsize
        self.stride = stride
        self.dilation = dilation
        self.k,self.pt = k,pt
        self.ws,self.wt = ws,wt
        self.batch_size = batch_size
        self.temp_opt = temp_opt
        if k <= 0:
            self.aggregation = None
        else:
            self.aggregation = N3AggregationBase(k, temp_opt=temp_opt)

    def forward(self, x, xe, ye, y=None, log_temp=None, flows=None):
        r"""
        :param x: database image
        :param xe: embedding of database image
        :param ye: embedding of query image
        :param y: query image, if None then y=x is assumed
        :param log_temp: optional log temperature image
        :return:
        """
        # -- assign if none --
        if self.aggregation is None:
            return y if y is not None else x
        if y is None:
            y = x
            ye = xe

        # -- add padding --
        x = F.pad(x,(1,1,1,1))
        xe = F.pad(xe,(1,1,1,1))
        ye = F.pad(ye,(1,1,1,1))
        y = F.pad(y,(1,1,1,1))

        # -- unpack --
        device = x.device
        t,c,h,w = x.shape
        vshape = (t,c*self.k,h,w)
        coords,dil = None,self.dilation
        stride = self.stride
        ps,k = self.patchsize,self.k
        ws,wt,pt = self.ws,self.wt,self.pt
        exact = False
        reflect_bounds = False
        rbounds = reflect_bounds
        use_adj = True
        adj = ps//2 if use_adj else 0
        # k_search = -1
        k_search = 224
        use_k = not(k_search == -1)
        use_search_abs = False
        full_ws = True
        only_full = False
        border_str = "reflect" if rbounds else "zero"
        # print("in: ",ws,wt,k,stride,dil,ps,pt,self.batch_size)
        # print("stride: " ,stride)

        # -- flows --
        fflow,bflow = get_flows(flows,(1,)+x.shape,x.device)

        # -- init fold --
        unfold = dnls.iUnfold(ps,coords,stride=stride,dilation=dil,
                              adj=adj,border=border_str,only_full=only_full)
        fold = dnls.iFoldz((1,)+vshape,coords,stride=stride,dilation=dil,
                          adj=adj,use_reflect=rbounds,only_full=only_full)

        # -- init search --
        adj = ps//2 if use_adj else 0
        # h0_off,w0_off,h1_off,w1_off = 1,1,1,1
        h0_off,w0_off,h1_off,w1_off = 0,0,0,0
        get_batch = dnls.utils.inds.get_query_batch
        search = dnls.search.init("l2_with_index",fflow, bflow, k_search,
                                  ps, pt, ws, wt,
                                  chnls=-1,dilation=dil,
                                  stride0=stride,stride1=stride,
                                  reflect_bounds=reflect_bounds,
                                  search_abs=use_search_abs,
                                  use_k=use_k,use_adj=use_adj,
                                  full_ws=full_ws,exact=exact,
                                  h0_off=h0_off,w0_off=w0_off,
                                  h1_off=h1_off,w1_off=w1_off,
                                  remove_self=True)

        # -- weighted patch sum --
        h_off,w_off = 0,0
        wpsum = dnls.reducers.WeightedPatchSumHeads(ps, pt, h_off=h_off,w_off=w_off,
                                                    dilation=dil, adj=adj, exact=exact,
                                                    reflect_bounds=reflect_bounds)

        # -- batching --
        # print("x.shape: ",x.shape)
        rm_pix = 0#dil*(ps-1)# if only_full else dil*(ps//2-1)
        nh = (h - rm_pix - 1)//stride + 1
        nw = (w - rm_pix - 1)//stride + 1
        ntotal_t = nh * nw
        ntotal = t * ntotal_t
        nbatch = self.batch_size
        if nbatch is None: nbatch = t*ntotal_t
        nbatch = min(nbatch,ntotal)
        # nbatch = ntotal
        nbatches = (ntotal-1) // nbatch + 1
        # print("x.shape: ",x.shape)

        # -- aggregation --
        if self.aggregation is None:
            return y if y is not None else x
        # print("x.shape: ",x.shape)
        # print("xe.shape: ",xe.shape)
        # print("ye.shape: ",ye.shape)

        # -- run for batches --
        for batch in range(nbatches):

            # -- batch info --
            # print(batch)
            qindex = min(nbatch * batch,ntotal)
            nbatch_i =  min(nbatch, ntotal - qindex)

            # -- queries --
            # qinds = get_batch(qindex,nbatch_i,stride,t,h,w,device)

            # -- exec --
            self.aggregation(x,xe,ye,log_temp,
                             search,unfold,fold,wpsum,
                             qindex,nbatch_i)

        # -- final steps --
        vid,zvid = fold.vid[0],fold.zvid[0]
        z = fold.vid / (zvid+1e-10)
        z = z.contiguous().view(t,k,c,h,w)
        z = z-y.view(t,1,c,h,w)
        z = z.view(t,k*c,h,w)

        # Concat with input
        z = th.cat([y, z], dim=1)
        z = z[...,1:-1,1:-1].contiguous()

        return z
