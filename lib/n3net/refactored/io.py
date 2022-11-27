

# -- misc --
import sys,os,copy
from pathlib import Path
from functools import partial
from easydict import EasyDict as edict

# -- torch --
import torch as th

# -- linalg --
import numpy as np

# -- modules --
from .n3net import N3Net

# -- misc imports --
from n3net.utils.misc import optional as _optional
from n3net.utils.model_utils import load_checkpoint,select_sigma,get_model_weights

# -- auto populate fields to extract config --
_fields = []
def optional_full(init,pydict,field,default):
    if not(field in _fields) and init:
        _fields.append(field)
    return _optional(pydict,field,default)

# -- model load wrapper --
def load_model(**kwargs):
    task = _optional(kwargs,"task","denoising")
    if task == "denoising":
        return load_model_deno(**kwargs)
    elif task == "sr":
        raise NotImplementedError("")
    else:
        raise ValueError(f"Uknown tasks [{task}]")

# -- load model --
def load_model_deno(**kwargs):

    # -- misc --
    init = _optional(kwargs,'__init',False) # purposefully weird key
    optional = partial(optional_full,init)
    device = optional(kwargs,'device','cuda:0')
    nfeatures_interm = optional(kwargs,"nfeatures_interm",8)
    ndncnn = optional(kwargs,"ndcnn",3)
    ntype =  optional(kwargs,"ntype","gaussian")
    model_name = optional(kwargs,"model_name","") # just add to model_cfg

    # -- io --
    sigma = optional(kwargs,"sigma",50.)
    task = _optional(kwargs,"task","denoising")

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

    # -- non-local options --
    k = optional(kwargs,'k',7)
    k = 7
    pt = optional(kwargs,'pt',1)
    ps = optional(kwargs,'ps',7)
    stride = optional(kwargs,'stride',5)
    dilation = optional(kwargs,'dilation',1)
    ws = optional(kwargs,'ws',-1)
    wt = optional(kwargs,'wt',0)
    batch_size = optional(kwargs,'bs',None)

    # -- io --
    pretrained_load = optional(kwargs,'pretrained_load',True)

    # -- end init --
    if init: return
    print("pretrained_load: ",pretrained_load)
    print("embedcnn_nplanes_out: ",embedcnn_nplanes_out)


    # -- args --
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

    # -- declare model --
    ninchannels=1
    noutchannels=1
    residual = False
    print(ws,wt,k,stride,dilation,ps,pt,batch_size)
    model = N3Net(ninchannels, noutchannels, nfeatures_interm, ndncnn,
                  residual, dncnn_opt, nl_temp_opt, embedcnn_opt,
                  ws=ws, wt=wt, k=k, stride=stride, dilation=dilation,
                  patchsize=ps, pt=pt, batch_size=batch_size)
    model = model.to(device)

    # -- load weights --
    if pretrained_load:

        # -- filename --
        fdir = Path(__file__).absolute().parents[0] / "../../../" # parent of "./lib"
        state_fn = get_model_weights(fdir,sigma,ntype)
        assert os.path.isfile(str(state_fn))
        print("Loading: ",state_fn)

        # -- fill weights --
        load_checkpoint(model,state_fn)

    # -- eval mode as default --
    model.eval()

    return model

# -- get all relevant model_cfg keys from entire cfg --
def extract_model_io(cfg):
    # -- auto populated fields --
    fields = _fields
    model_cfg = {}
    for field in fields:
        if field in cfg:
            model_cfg[field] = cfg[field]
    return edict(model_cfg)

# -- run to populate "_fields" --
load_model(__init=True)
