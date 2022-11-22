

# -- misc --
import sys,os,copy
from pathlib import Path
from easydict import EasyDict as edict

# -- torch --
import torch as th

# -- linalg --
import numpy as np

# -- modules --
from .n3net import N3Net

# -- misc imports --
from n3net.utils.misc import optional
from n3net.utils.model_utils import load_checkpoint,select_sigma,get_model_weights

def load_model(cfg=None):
    mtype = optional(cfg,"model_type","denoising")
    if mtype == "denoising":
        return load_model_deno(cfg)
    elif mtype == "sr":
        raise NotImplementedError("")
    else:
        raise ValueError("")

def load_model_deno(cfg):

    # -- misc --
    cfg = edict(cfg)
    sigma = cfg.sigma
    device = optional(cfg,'device','cuda:0')
    patchsize = optional(cfg,'patchsize',80)
    nfeatures_interm = optional(cfg,"nfeatures_interm",8)
    ndncnn = optional(cfg,"ndcnn",3)
    ntype =  optional(cfg,"ntype","gaussian")

    # -- non-local --
    nl_k = optional(cfg,'nl_k',7)
    nl_ps = optional(cfg,'nl_patchsize',10)
    nl_stride = optional(cfg,'nl_stride',5)

    # -- non-local [temp] --
    nl_temp_avgpool = optional(cfg,'nl_temp_avgpool',"true") == "true"
    nl_temp_distance_bn = optional(cfg,'nl_temp_distance_bn',"true") == "true"
    nl_temp_external_temp = optional(cfg,'nl_temp_external_temp',"true") == "true"
    nl_temp_temp_bias = optional(cfg,'nl_temp_temp_bias',0.1)


    # -- dncnn --
    dncnn_bn = optional(cfg,"dncnn_bn","true") == "true"
    dncnn_depth = optional(cfg,"dncnn_depth",6)
    dncnn_kernel = optional(cfg,"dncnn_kernel",3)
    dncnn_features = optional(cfg,"dncnn_features",64)

    # -- embedding --
    embedcnn_features = optional(cfg,"embedcnn_features",64)
    embedcnn_depth = optional(cfg,"embedcnn_depth",3)
    embedcnn_kernel = optional(cfg,"embedcnn_kernel",3)
    embedcnn_nplanes_out = optional(cfg,"embedcnn_nplanes_out",8)
    embedcnn_bn = optional(cfg,"embedcnn_bn","true") == "true"

    # -- relevant configs --
    fwd_mode = optional(cfg,'fwd_mode',"dnls_k")
    ws = optional(cfg,'ws',-1)
    wt = optional(cfg,'wt',0)
    k = optional(cfg,'k',-1)
    sb = optional(cfg,'sb',None)

    # -- args --
    ninchannels=1
    noutchannels=1
    nl_temp_opt = dict(
        avgpool = nl_temp_avgpool,
        distance_bn = nl_temp_distance_bn,
        external_temp = nl_temp_external_temp,
        temp_bias = nl_temp_temp_bias)
    embedcnn_opt = dict(
        features    = embedcnn_features,
        depth       = embedcnn_depth,
        kernel      = embedcnn_kernel,
        nplanes_out = embedcnn_nplanes_out,
        bn          = embedcnn_bn)
    n3block_opt = dict(
        k=nl_k,
        patchsize=nl_ps,
        stride=nl_stride,
        temp_opt=nl_temp_opt,
        embedcnn_opt=embedcnn_opt)
    dncnn_opt = dict(
        bn = dncnn_bn,
        depth = dncnn_depth,
        features = dncnn_features,
        kernel = dncnn_kernel,
        residual = True)

    # -- init model --
    model = N3Net(ninchannels, noutchannels, nfeatures_interm,
                  nblocks=ndncnn, block_opt=dncnn_opt,
                  nl_opt=n3block_opt, residual=False)
    model = model.to(device)

    # -- load weights --
    fdir = Path(__file__).absolute().parents[0] / "../../../" # parent of "./lib"
    state_fn = get_model_weights(fdir,sigma,ntype)
    print("state_fn: ",state_fn)
    assert os.path.isfile(str(state_fn))

    # -- fill weights --
    load_checkpoint(model,state_fn)

    # -- eval mode as default --
    model.eval()

    return model


