

# -- misc --
import sys,os,copy
from pathlib import Path

# -- torch --
import torch as th

# -- linalg --
import numpy as np

# -- modules --
from .n3net import N3Net

# -- misc imports --
from n3net.utils.misc import optional
from n3net.utils.model_utils import load_checkpoint,select_sigma,get_model_weights

def load_model(mtype,sigma,cfg=None):
    if mtype == "denoising":
        return load_model_deno(sigma,cfg)
    elif mtype == "sr":
        raise NotImplementedError("")
    else:
        raise ValueError("")

def load_model_deno(sigma,kwargs):

    # -- misc --
    device = optional(kwargs,'device','cuda:0')
    nfeatures_interm = optional(kwargs,"nfeatures_interm",8)
    ndncnn = optional(kwargs,"ndcnn",3)
    ntype =  optional(kwargs,"ntype","gaussian")

    # -- non-local [temp] --
    nl_temp_avgpool = optional(kwargs,'nl_temp_avgpool',"true") == "true"
    nl_temp_distance_bn = optional(kwargs,'nl_temp_distance_bn',"true") == "true"
    nl_temp_external_temp = optional(kwargs,'nl_temp_external_temp',"true") == "true"
    nl_temp_temp_bias = optional(kwargs,'nl_temp_temp_bias',0.1)


    # -- dncnn --
    dncnn_bn = optional(kwargs,"dncnn_bn","true") == "true"
    dncnn_depth = optional(kwargs,"dncnn_depth",6)
    dncnn_kernel = optional(kwargs,"dncnn_kernel",3)
    dncnn_features = optional(kwargs,"dncnn_features",64)

    # -- embedding --
    embedcnn_features = optional(kwargs,"embedcnn_features",64)
    embedcnn_depth = optional(kwargs,"embedcnn_depth",3)
    embedcnn_kernel = optional(kwargs,"embedcnn_kernel",3)
    embedcnn_nplanes_out = optional(kwargs,"embedcnn_nplanes_out",8)
    embedcnn_bn = optional(kwargs,"embedcnn_bn","true") == "true"

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
    dncnn_opt = dict(
        bn = dncnn_bn,
        depth = dncnn_depth,
        features = dncnn_features,
        kernel = dncnn_kernel,
        residual = True)

    # -- non-local options --
    k = optional(kwargs,'k',7)
    pt = optional(kwargs,'pt',1)
    ps = optional(kwargs,'ps',7)
    stride = optional(kwargs,'stride',5)
    dilation = optional(kwargs,'dilation',1)
    ws = optional(kwargs,'ws',-1)
    wt = optional(kwargs,'wt',0)
    batch_size = optional(kwargs,'bs',None)
    residual = False

    # -- declare model --
    print(ws,wt,k,stride,dilation,ps,pt,batch_size)
    model = N3Net(ninchannels, noutchannels, nfeatures_interm, ndncnn,
                  residual, dncnn_opt, nl_temp_opt, embedcnn_opt,
                  ws=ws, wt=wt, k=k, stride=stride, dilation=dilation,
                  patchsize=ps, pt=pt, batch_size=batch_size)
    model = model.to(device)

    # -- load weights --
    fdir = Path(__file__).absolute().parents[0] / "../../../" # parent of "./lib"
    state_fn = get_model_weights(fdir,sigma,ntype)
    assert os.path.isfile(str(state_fn))
    print("state_fn: ",state_fn)

    # -- fill weights --
    load_checkpoint(model,state_fn)

    # -- eval mode as default --
    model.eval()

    return model


