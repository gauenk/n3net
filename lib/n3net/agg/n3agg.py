"""

Aggregation using the stnls method

"""

import stnls
import n3net
import torch as th
from einops import rearrange
from ..shared_model import ops
from ..augmented import vid_to_raster_inds

class N3Agg():

    def __init__(self,ps,stride0):
        self.ps = ps
        self.stride0 = stride0
        coords = None
        dil = 1
        adj = True
        only_full = False
        rbounds = False
        border_str = "reflect" if rbounds else "zero"
        self.unfold = stnls.iUnfold(ps,coords,stride=stride0,dilation=dil,
                                   adj=adj,border=border_str,only_full=only_full)

    def __call__(self,vid,dists,inds):
        print("vid.shape,inds.shape: ",vid.shape,inds.shape)
        if vid.ndim == 5:
            return self.run_batch(vid,dists,inds)
        else:
            return self.run(vid,dists,inds)

    def run_batch(self,vid,dists,inds):
        B = vid.shape[0]
        patches = []
        for b in range(B):
            patches_b = self.run(vid[b],dists[b],inds[b])
            patches.append(patches_b)
        patches = th.stack(patches)
        return patches

    def run(self,vid,dists,inds):

        # -- get indices --
        ps = self.ps
        dev = vid.device
        t,c,iH,iW = vid.shape
        # print("vid.shape,inds.shape,dists.shape: ",vid.shape,inds.shape,dists.shape)
        I = vid_to_raster_inds(inds,iH,iW,self.stride0,dev)
        # print(I.min(),I.max())

        # -- unfold and opt --
        x_patches = self.unfold(vid)
        dists = rearrange(dists,"thw s -> 1 thw s 1")
        s = dists.shape[2] # conceptually, the "k"
        x_patches = rearrange(x_patches,'thw 1 1 c ph pw -> 1 thw (c ph pw)')
        # print("x_patches.shape,I.shape,dists.shape: ",x_patches.shape,I.shape,dists.shape)
        z_patches = ops.indexed_matmul_2_efficient(x_patches, dists, I, chunk_size=s)
        shape_str = 'b q (c ph pw) k -> b q 1 1 (k c) ph pw'
        patches = rearrange(z_patches,shape_str,ph=ps,pw=ps)
        return patches
