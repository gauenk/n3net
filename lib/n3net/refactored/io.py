

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
def load_model(cfg):
    task = _optional(cfg,"task","denoising")
    if task == "denoising":
        return load_model_deno(cfg)
    elif task == "sr":
        raise NotImplementedError("")
    else:
        raise ValueError(f"Uknown tasks [{task}]")

# -- load model --
def load_model_deno(cfg):

    # -- misc --
    init = _optional(cfg,'__init',False) # purposefully weird key
    optional = partial(optional_full,init)
    device = optional(cfg,'device','cuda:0')
    nfeatures_interm = optional(cfg,"nfeatures_interm",8)
    ndncnn = optional(cfg,"ndcnn",3)
    ntype =  optional(cfg,"ntype","gaussian")
    model_name = optional(cfg,"model_name","") # just add to model_cfg

    # -- io --
    sigma = optional(cfg,"sigma",50.)
    task = _optional(cfg,"task","denoising")

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

    # -- non-local options --
    k = optional(cfg,'k',7)
    k = 7
    pt = optional(cfg,'pt',1)
    ps = optional(cfg,'ps',7)
    stride = optional(cfg,'stride',5)
    dilation = optional(cfg,'dilation',1)
    ws = optional(cfg,'ws',-1)
    wt = optional(cfg,'wt',0)
    batch_size = optional(cfg,'bs',None)

    # -- io --
    pretrained_load = optional(cfg,'pretrained_load',True)

    # -- end init --
    if init: return
    # print("pretrained_load: ",pretrained_load)
    # print("embedcnn_nplanes_out: ",embedcnn_nplanes_out)

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
load_model({"__init":True})
