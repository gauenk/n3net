
import math
from math import log

import numpy as np
import torch as th
from einops import rearrange,repeat

import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import dnls
import n3net
from dnls.utils.pads import comp_pads
from n3net.utils.misc import get_flows
from .non_local import NeuralNearestNeighbors
from .non_local import vid_index_neighbours,vid_to_raster_inds

class N3AggregationBase(nn.Module):
    r"""
    Domain agnostic base class for computing neural nearest neighbors
    """
    def __init__(self, k, use_cts_topk, dist_scale, temp_opt={}):
        r"""
        :param k: Number of neighbor volumes to compute
        :param temp_opt: options for handling temperatures, see `NeuralNearestNeighbors`
        """
        super(N3AggregationBase, self).__init__()
        self.k = k
        self.temp_opt = temp_opt
        self.nnn = NeuralNearestNeighbors(k, temp_opt=temp_opt)
        self.dist_scale = dist_scale
        self.use_cts_topk = use_cts_topk

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
        # print("xe.shape,ye.shape: ",xe.shape,ye.shape)
        dists,inds = search(xe[None,:],qindex,nbatch_i,ye[None,:])
        dists,inds = dists[0],inds[0]
        dists = -dists
        # print("dists.shape: ",dists.shape)
        # print(xe.max(),xe.min())
        # print(ye.max(),ye.min())

        # -- log_temp patches --
        if self.use_cts_topk:
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

            # -- viz --
            # print(W[0,0,:10])
            # print(W[:,0,:10])
            # for i in range(7):
            #     print(i,W[i,250,:10])
            # for i in range(7):
            #     print(i,W[i,250:270,:2])
            # S = F.softmax(dists,1)
            # print(S[0,0,:10])
            # print(S[0,250,:10])
            # print(S[0,250:270,:2])
            # print("dists.shape: ",dists.shape)
            # print("S.shape: ",dists.shape)
            # print("W.shape: ",W.shape)
        else:
            # print(dists[:5,:10])
            scale = self.dist_scale
            # print("scale: ",scale)
            dists = dists[None,:]
            # print("dists.mean((0,1))[:10]: ",dists.mean((0,1))[:10])
            W = F.softmax(scale*dists,2)
            # print(W[0])
        # print("W.shape: ",W.shape)

        nheads,b,_ = W.shape
        c = x.shape[-3]
        # ps = wpsum.ps
        # print("inds.shape: ",inds.shape)
        # print("x.shape,W.shape,inds.shape: ",x.shape,W.shape,inds.shape)
        # z_patches = wpsum(x[None,:],W[None,:],inds[None,:]) # b self.k c h w
        # print("z_patches.shape: ",z_patches.shape)
        ps = search.ps

        use_unfold = True
        if use_unfold:

            # -- get indices --
            dev = x.device
            t,c,iH,iW = x.shape
            stride = search.stride0#5
            I = vid_to_raster_inds(inds,iH,iW,stride,dev)

            # -- unfold and opt --
            x_patches = unfold(x)
            # print("W.shape: ",W.shape)
            # print("x_patches.shape: ",x_patches.shape)
            # W = rearrange(W,"k thw s -> 1 thw s k")
            W = rearrange(W,"k thw s -> 1 thw s k")
            s = W.shape[2] # conceptually, the "k"
            x_patches = rearrange(x_patches,'thw 1 1 c ph pw -> 1 thw (c ph pw)')
            z_patches = n3net.ops.indexed_matmul_2_efficient(x_patches, W,
                                                             I, chunk_size=s)
            shape_str = 'b q (c ph pw) k -> b q 1 1 (k c) ph pw'
            z_patches = rearrange(z_patches,shape_str,ph=ps,pw=ps)
        else:
            z_patches = wpsum(x[None,:],W[None,:],inds[None,:]) # b self.k c h w
            shape_str = 'b k q c ph pw -> b q 1 1 (k c) ph pw'
            z_patches = rearrange(z_patches,shape_str,ph=ps,pw=ps)
            fold(z_patches,qindex)


        # -- fold into video --
        # print("z_patches.shape: ",z_patches.shape)
        fold(z_patches,qindex)

class N3Aggregation2D(nn.Module):
    r"""
    Computes neural nearest neighbors for image data based on extracting patches
    in strides.
    """
    def __init__(self, k=7, patchsize=10, stride=1, dilation=1,
                 ws=29, wt=0, pt=1, batch_size=None, use_cts_topk=False,
                 dist_scale=1.,nbwd=1, rbwd=True, temp_opt={}):
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
        self.use_cts_topk = use_cts_topk
        self.nbwd = nbwd
        self.rbwd = rbwd
        self.temp_opt = temp_opt
        self.dist_scale = dist_scale
        if k <= 0:
            self.aggregation = None
        else:
            self.aggregation = N3AggregationBase(k, use_cts_topk,
                                                 dist_scale, temp_opt=temp_opt)

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

        # -- flows --
        fflow,bflow = get_flows(flows,(1,)+x.shape,x.device)
        # fflow,bflow = fflow[None,:],bflow[None,:]
        # I = self.get_vid_index(x.shape)

        # -- add padding --
        x = F.pad(x,(1,1,1,1))
        xe = F.pad(xe,(1,1,1,1))
        ye = F.pad(ye,(1,1,1,1))
        y = F.pad(y,(1,1,1,1))
        fflow = F.pad(fflow,(1,1,1,1))
        bflow = F.pad(bflow,(1,1,1,1))

        # -- unpack --
        k_shape = 7 if self.use_cts_topk else 1
        device = x.device
        t,c,h,w = x.shape
        vshape = (t,c*k_shape,h,w)
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
        # k_search = 224
        k_search = 224 if self.use_cts_topk else self.k
        use_k = not(k_search == -1)
        use_search_abs = False
        full_ws = True
        only_full = False
        border_str = "reflect" if rbounds else "zero"
        remove_self = True#self.use_cts_topk

        # -- init fold --
        unfold = dnls.iUnfold(ps,coords,stride=stride,dilation=dil,
                              adj=adj,border=border_str,only_full=only_full)
        fold = dnls.iFoldz((1,)+vshape,coords,stride=stride,dilation=dil,
                          adj=adj,use_reflect=rbounds,only_full=only_full)

        # -- init search --
        # h0_off,w0_off,h1_off,w1_off = 1,1,1,1
        # h0_off,w0_off,h1_off,w1_off = 1,1,1,1
        # adj = 0
        # use_adj = False
        h0_off,w0_off,h1_off,w1_off = 0,0,0,0
        # h0_off,w0_off,h1_off,w1_off = -2,-2,-2,-2
        # h0_off,w0_off,h1_off,w1_off = -3,-3,-3,-3
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
                                  remove_self=remove_self,
                                  rbwd=self.rbwd,nbwd=self.nbwd)

        # -- weighted patch sum --
        h_off,w_off = 0,0
        # adj = 0
        # h_off,w_off = -3,-3
        wpsum = dnls.reducers.WeightedPatchSumHeads(ps, pt, h_off=h_off,w_off=w_off,
                                                    dilation=dil, adj=adj,
                                                    exact=exact,rbwd=True,
                                                    reflect_bounds=reflect_bounds)

        # -- batching --
        # rm_pix = 0#dil*(ps-1)# if only_full else dil*(ps//2-1)
        rm_pix = dil*(ps-1) if only_full else 0
        nh = (h - rm_pix - 1)//stride + 1
        nw = (w - rm_pix - 1)//stride + 1
        ntotal_t = nh * nw
        ntotal = t * ntotal_t
        nbatch = self.batch_size
        if nbatch is None: nbatch = t*ntotal_t
        nbatch = min(nbatch,ntotal)
        # nbatch = ntotal
        nbatches = (ntotal-1) // nbatch + 1

        # -- aggregation --
        if self.aggregation is None:
            return y if y is not None else x

        # -- run for batches --
        assert nbatches == 1
        for batch in range(nbatches):

            # -- batch info --
            qindex = min(nbatch * batch,ntotal)
            nbatch_i =  min(nbatch, ntotal - qindex)

            # -- exec --
            self.aggregation(x,xe,ye,log_temp,
                             search,unfold,fold,wpsum,
                             qindex,nbatch_i)

        # -- final steps --
        vid,zvid = fold.vid[0],fold.zvid[0]
        z = vid / zvid
        if th.any(th.isnan(z)).item():
            print("isnan(z)")
            exit(0)
        # print("[a] z.shape: ",z.shape)
        z = z.contiguous().view(t,k_shape,c,h,w)
        # print("[b] z.shape: ",z.shape)
        z = z-y.view(t,1,c,h,w)
        # print("[c] z.shape: ",z.shape)
        z = z.view(t,k_shape*c,h,w)
        # print("[d] z.shape: ",z.shape)

        # Concat with input
        # print("z.shape: ",z.shape)
        # print("y.shape: ",y.shape)

        z = th.cat([y, z], dim=1)
        z = z[...,1:-1,1:-1].contiguous()

        # print("z.shape: ",z.shape)
        return z
