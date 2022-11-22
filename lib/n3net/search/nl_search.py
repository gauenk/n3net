"""
The n3net search function

"""

# -- python --
import torch as th
import torch.nn as nn

# -- n3net functions --
import n3net.shared_model.ops as ops
from ..original.non_local import index_neighbours,index_neighbours_cache
from ..original.non_local import compute_distances

def get_topk(l2_vals,l2_inds,k):

    # -- reshape exh --
    b,nq = l2_vals.shape[:2]
    l2_vals = l2_vals.view(b,nq,-1)
    l2_inds = l2_inds.view(b,nq,-1)
    # l2_inds = l2_inds.view(b,nq,-1,3)

    # -- init --
    vals = th.zeros((b,nq,k),dtype=l2_vals.dtype)
    inds = th.zeros((b,nq,k),dtype=l2_inds.dtype)
    # inds = th.zeros((b,nq,k,3),dtype=l2_inds.dtype)

    # -- take mins --
    order = th.argsort(l2_vals,dim=2,descending=False)
    vals[:,:nq,:] = th.gather(l2_vals,2,order[:,:,:k])
    inds[:,:nq,:] = th.gather(l2_inds,2,order[:,:,:k])
    # for i in range(inds.shape[-1]):
    #     inds[:nq,:,i] = th.gather(l2_inds[:,:,i],1,order[:,:k])

    return vals,inds

class NLSearch():

    def __init__(self,k=7, ps=7, ws=8, nheads=1, chnls=-1, dilation=1,
                 stride0=1, stride1=1, index_reset=True, include_self=False,
                 use_k=True):
        self.k = k
        self.ps = ps
        self.nheads = nheads
        self.ws = ws
        self.chnls = chnls
        self.dilation = dilation
        self.stride0 = stride0
        self.stride1 = stride1
        self.index_reset = index_reset
        self.include_self = include_self
        self.padding = (ps-1)//2
        self.use_k = use_k
        self.index_reset = index_reset

    def __call__(self,vid,foo=0,bar=0):
        if vid.ndim == 5:
            return self.search_batch(vid)
        else:
            return self.search(vid)

    def search_batch(self,vid):
        dists,inds = [],[]
        B = vid.shape[0]
        for b in range(B):
            _dists,_inds = self.search(vid[b])
            dists.append(_dists)
            inds.append(_inds)
        dists = th.stack(dists)
        inds = th.stack(inds)
        return dists,inds

    def search(self,vid):

        # -- patchify --
        padding = None
        xe = vid
        x_patch, padding = ops.im2patch(vid, self.ps, self.stride0,
                                        None, returnpadding=True)
        # xe_patch = ops.im2patch(xe, self.ps, self.stride0, self.padding)
        xe_patch = ops.im2patch(xe, self.ps, self.stride0, None)

        # -- self attn --
        ye_patch = xe_patch

        # -- indexing function --
        inds = index_neighbours(xe_patch, ye_patch, self.ws,
                                exclude_self=not(self.include_self))
        # print("inds.shape: ",inds.shape,self.ws)
        if self.index_reset:
            index_neighbours_cache.clear()

        # -- reshaping --
        b,c,p1,p2,n1,n2 = x_patch.shape
        _,ce,e1,e2,m1,m2 = ye_patch.shape
        _,_,o = inds.shape
        _,_,H,W = vid.shape
        n = n1*n2; m=m1*m2; f=c*p1*p2; e=ce*e1*e2
        # print((b,n,e),(b,m,e))
        xe = xe_patch.permute(0,4,5,1,2,3).contiguous().view(b,n,f)
        ye = ye_patch.permute(0,4,5,1,2,3).contiguous().view(b,m,e)

        # -- compute attn --
        dists = compute_distances(xe, ye, inds, False)

        # -- topk --
        if self.use_k:
            dists,inds = get_topk(dists,inds,self.k)

        return dists,inds



