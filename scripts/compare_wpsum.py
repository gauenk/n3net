"""

Comparing the WPSUM from DNLS and N3Net

"""

# -- misc --
import os,copy
dcopy = copy.deepcopy
import math,tqdm
import pprint,random
pp = pprint.PrettyPrinter(indent=4)

# -- linalg --
import numpy as np
import torch as th
from einops import rearrange,repeat

# -- data mngmnt --
from pathlib import Path
from easydict import EasyDict as edict

# -- data --
import data_hub

# -- caching results --
import cache_io

# -- network --
import n3net
from n3net.utils import optional
from n3net.utils.timer import ExpTimer,TimeIt
import n3net.shared_model.ops as n3net_ops
from n3net.original import index_neighbours

# -- wpsum --
from stnls import reducers as stnls_r

def prepare_data_stnls(_y,_x,_I,H,W):
    print("_y.shape: ",_y.shape)
    print("_x.shape: ",_x.shape)
    print("_I.shape: ",_I.shape)
    T,hw = _I.shape[:2]
    _y = rearrange(_y,'t hw s k -> k (t hw) s')
    _y = _y.contiguous()

    stride = 5
    nH  = (H-1)//stride#+1
    nW  = (W-1)//stride#+1
    # print(nH,nW)

    # for i in range(10):
    #     print(_I[0,i])

    I = []
    for t in range(T):
        # -- "1-d -> 3-d" --
        w = th.remainder(_I[t], nW)*stride
        _div = th.div(_I[t], nW, rounding_mode="floor")
        h = th.remainder(_div, nH)*stride
        # t = th.div(_div, H, rounding_mode="floor")
        t = t * th.ones_like(h).type(th.int32)
        I_t = th.stack([t,h,w],-1)
        I.append(I_t)
    I = th.cat(I,0).type(th.int32)
    # order = th.argsort(I[0,:,2])
    # print(I[0])
    # for j in range(3):
    #     I[0,:,j] = th.gather(I[0,:,j],0,order)
    # order = th.argsort(I[0,:,1])
    # for j in range(3):
    #     I[0,:,j] = th.gather(I[0,:,j],0,order)
    # order = th.argsort(I[0,:,2])
    # for j in range(3):
    #     I[0,:,j] = th.gather(I[0,:,j],0,order)


    # print(I[0])
    # print("I.shape: ",I.shape)
    # print("_y.shape: ",_y.shape)
    # exit()

    # for i in range(3):
    #     print(I[i])

    return _y,_x,I

def prepare_data(_y,_x,_I,H,W,name):
    if name == "stnls":
        _y,_x,_I = prepare_data_stnls(_y,_x,_I,H,W)
    else:
        _x = rearrange(_x,'t c ph pw nH nW -> t (nH nW) (c ph pw)')
        _x = _x.contiguous()
        _y = _y.clone()
        _x = _x.clone()
        # _y,_x,_I = prepare_data_n3net(_y,_x,_I,H,W)
    y,x = _y.clone(),_x.clone()
    y = y.requires_grad_(True)
    x = x.requires_grad_(True)
    return y,x,_I

def run_exp(_cfg):

    # -- init --
    cfg = dcopy(_cfg)
    cache_io.exp_strings2bools(cfg)
    timer = ExpTimer()

    # -- unpack --
    s = cfg.s
    ps,pt = cfg.ps,cfg.pt
    o = s**2-1
    k,n = cfg.k,cfg.n
    exact = cfg.exact
    rbwd = cfg.rbwd
    nbwd = cfg.nbwd
    stride = cfg.stride
    h_off,w_off = -3,-3
    dil = 1
    reflect_bounds = False
    adj = 0#ps//2

    # -- load video --
    data,loaders = data_hub.sets.load(cfg)
    frame_start = optional(cfg,"frame_start",0)
    frame_end = optional(cfg,"frame_end",-1)
    indices = data_hub.filter_subseq(data[cfg.dset],cfg.vid_name,frame_start,frame_end)
    vid = data[cfg.dset][0]['noisy'].to(cfg.device)/255.
    # vid[...] = 1.
    # r = (f-1)/vid.shape[0]+1
    # vid = repeat(vid,'t c h w -> t (r c) h w',r=r)[:,:f]
    print("vid.shape: ",vid.shape)
    T,C,H,W = vid.shape
    b =  T

    # -- init sample data --
    x_patches = n3net_ops.im2patch(vid, ps, stride, padding=None, returnpadding=False)
    y_patches = x_patches.clone()
    I = index_neighbours(x_patches,y_patches,s,exclude_self=True)
    m1,m2 = x_patches.shape[-2:]
    m = m1*m2
    Wmat = th.rand((b,m,o,k),dtype=th.float32,device=cfg.device)
    # Wmat = th.ones((b,m,o,k),dtype=th.float32,device=cfg.device)
    # print("x_patches.shape: ",x_patches.shape)
    # print("Wmat.shape: ",Wmat.shape)
    # print("I.shape: ",I.shape)

    # -- run stnls --
    Wmat_stnls = rearrange(Wmat,'t hw s k -> k (t hw) s').contiguous()
    y_stnls,x_stnls,I_stnls = prepare_data(Wmat,vid,I,H,W,"stnls")
    wpsum = stnls_r.WeightedPatchSumHeads(ps, pt, h_off=h_off,w_off=w_off,
                                         dilation=dil, adj=adj, exact=exact,
                                         rbwd=rbwd,nbwd=nbwd,
                                         reflect_bounds=reflect_bounds)
    print("Wmat_stnls.shape: ",Wmat_stnls.shape)
    # print("vid.shape, Wmat.shape, I.shape: ",vid.shape, Wmat.shape, I.shape)
    if cfg.warmup:
        out = wpsum(vid[None,:],Wmat_stnls[None,:],I_stnls[None,:])
        th.cuda.synchronize()
    with TimeIt(timer,"stnls_fwd"):
        out_stnls = wpsum(x_stnls[None,:],y_stnls[None,:],I_stnls[None,:])
    out_stnls = rearrange(out_stnls,'1 k (t hw) c ph pw -> t hw (c ph pw) k',t=b)

    # -- run n3net --
    vid_n3 = vid.requires_grad_(True)
    x_n3 = n3net_ops.im2patch(vid_n3, ps, stride, padding=None, returnpadding=False)
    x_n3 = rearrange(x_n3,'t c ph pw nH nW -> t (nH nW) (c ph pw)')
    x_n3 = x_n3.contiguous()

    # y_n3 = x_n3.clone()
    x_n3_te = n3net_ops.im2patch(vid, ps, stride, padding=None, returnpadding=False)
    x_n3_te = rearrange(x_n3_te,'t c ph pw nH nW -> t (nH nW) (c ph pw)')
    x_n3_te = x_n3_te.contiguous()
    y_n3 = Wmat.requires_grad_(True)
    # y_n3_te = x_n3_te.clone()
    # y_n3,x_n3,I_n3 = prepare_data(Wmat,x_patches,I,H,W,"n3")
    # _,x_n3_te,_ = prepare_data(Wmat,x_patches,I,H,W,"n3")
    # print("y_n3.shape: ",y_n3.shape)
    # print("x_n3.shape: ",x_n3.shape)
    # print("I_n3.shape: ",I_n3.shape)
    wpsum = n3net_ops.indexed_matmul_2_efficient
    if cfg.warmup:
        out = wpsum(x_n3_te,Wmat,I)
        th.cuda.synchronize()
    with TimeIt(timer,"n3net_fwd"):
        out_n3 = wpsum(x_n3,y_n3,I)

    print("out_n3.shape: ",out_n3.shape)

    # -- compare --
    cmp_fwd = th.abs(out_n3 - out_stnls).mean()

    # -- create grad --
    # grad = th.ones_like(out_n3)
    grad = th.randn_like(out_n3)

    # -- backward stnls --
    with TimeIt(timer,"stnls_bwd"):
        th.autograd.backward(out_stnls,grad)

    # -- backward n3net --
    with TimeIt(timer,"n3net_bwd"):
        th.autograd.backward(out_n3,grad)

    # -- cmp --
    cmps = {"x":(vid_n3,x_stnls),
            "y":(y_n3,y_stnls)}
    cmp_bwd = {}
    for name,pair in cmps.items():
        if name == "y":
            p0,p1 = pair[0],pair[1]
            p1_grad = p1.grad
            p0_grad = rearrange(p0.grad,'t hw s k -> k (t hw) s')
        else:
            p0,p1 = pair[0],pair[1]
            p0_grad = p0.grad
            p1_grad = p1.grad
            print(p0_grad[0,0,:3,:3])
            print(p1_grad[0,0,:3,:3])
            print(p0_grad[0,0,32:35,32:35])
            print(p1_grad[0,0,32:35,32:35])
        # print(name,"pair[0].shape,pair[1].shape: ",p0_grad.shape,p1_grad.shape)
        diff_pair = th.abs(p0_grad - p1_grad).mean().item()
        key_name = "cmp_bwd_%s" % name
        cmp_bwd[key_name] = diff_pair

    # -- create report --
    results = edict()
    results.cmp_fwd = cmp_fwd.item()
    for time_key,time_val in timer.items():
        results[time_key] = time_val
    for cmp_key,cmp_val in cmp_bwd.items():
        results[cmp_key] = cmp_val
    print(cmp_bwd)

    return results

def main():

    # -- start info --
    verbose = True
    pid = os.getpid()
    print("PID: ",pid)

    # -- get cache --
    cache_dir = ".cache_io"
    cache_name = "test_rgb_net_rebuttle_testing" # current!
    cache = cache_io.ExpCache(cache_dir,cache_name)
    cache.clear()

    # -- init cfg --
    cfg = edict()
    cfg.nframes = 3
    cfg.device = "cuda:0"
    cfg.s = 15
    cfg.ps = 10
    cfg.pt = 1
    cfg.stride = 5
    cfg.dname = "set8"
    cfg.dset = "te"
    cfg.vid_name = "motorbike"
    cfg.sigma = 30
    cfg.isize = "256_256"
    # cfg.isize = "156_156"
    # cfg.isize = "128_128"
    # cfg.isize = "96_96"
    cfg.cropmode = "center"
    cfg.warmup = True

    # -- config grid --
    n = [100]
    k = [7]
    f = [64]
    exact = ["false"]
    # exact = ["true"]
    # rbwd = ["false"]
    # nbwd = [1]
    rbwd = ["true","false"]
    nbwd = [1,5]#,10,20]
    expl = {"n":n,"k":k,"f":f,"exact":exact,"rbwd":rbwd,"nbwd":nbwd}
    exps = cache_io.mesh_pydicts(expl) # create mesh
    cache_io.append_configs(exps,cfg)

    #
    # -=-=- Run Exps -=-=-
    #

    nexps = len(exps)
    print(nexps)
    for exp_num,exp in enumerate(exps):

        # -- info --
        if verbose:
            print("-="*25+"-")
            print(f"Running experiment number {exp_num+1}/{nexps}")
            print("-="*25+"-")
            pp.pprint(exp)

        # -- logic --
        uuid = cache.get_uuid(exp) # assing ID to each Dict in Meshgrid
        # cache.clear_exp(uuid)
        # if exp.model_name != "refactored":
        #     cache.clear_exp(uuid)
        results = cache.load_exp(exp) # possibly load result
        if results is None: # check if no result
            exp.uuid = uuid
            results = run_exp(exp)
            cache.save_exp(uuid,exp,results) # save to cache

    # -- load results --
    records = cache.load_flat_records(exps)
    print(records)
    for col in records.columns:
        if "time" in col or "cmp" in col or "nbwd" == col or "rbwd" == col:
            print(records[col])


if __name__ == "__main__":
    main()
