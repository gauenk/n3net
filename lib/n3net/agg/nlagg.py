"""

Aggregation using the stnls method

"""

import stnls
from einops import rearrange

def init_agg(ps):
    pt = 1
    h_off = 0
    w_off = 0
    dil = 1
    adj = True
    exact = False
    rbwd = False
    reflect_bounds = False
    wpsum = stnls.reducers.WeightedPatchSumHeads(ps, pt, h_off=h_off,w_off=w_off,
                                                dilation=dil, adj=adj,
                                                exact=exact,rbwd=rbwd,
                                                reflect_bounds=reflect_bounds)
    return wpsum

class NLAgg():

    def __init__(self,ps):
        self.stride0 = -1
        self.ps = ps
        self.agg = init_agg(ps)

    def __call__(self,vid,dists,inds):
        ps = self.ps
        patches = self.agg(vid[None,:],dists[None,:],inds[None,:])
        shape_str = 'b k q c ph pw -> b q 1 1 (k c) ph pw'
        patches = rearrange(patches,shape_str,ph=ps,pw=ps)
        return patches
