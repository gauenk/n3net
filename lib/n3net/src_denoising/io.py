

# -- misc --
import sys,os,copy
from pathlib import Path

# -- torch --
import torch as th

# -- linalg --
import numpy as np

# -- modules --
from .models.n3net import N3Net

# -- misc imports --
from n3net.utils.misc import optional,select_sigma
from n3net.utils.model_utils import load_checkpoint

def load_model(sigma,**kwargs):

    # -- misc --
    device = optional(kwargs,'device','cuda:0')
    patchsize = optional(kwargs,'patchsize',80)
    nfeatures_interm = optional(kwargs,"nfeatures_interm",8)
    ndncnn = optional(kwargs,"ndcnn",3)

    # -- non-local --
    nl_k = optional(kwargs,'nl_k',7)
    nl_ps = optional(kwargs,'nl_patchsize',10)
    nl_stride = optional(kwargs,'nl_stride',5)

    # -- non-local [temp] --
    nl_temp_avgpool = optional(kwargs,'nl_temp_avgpool',"true") == "true"
    nl_temp_distance_bn = optional(kwargs,'nl_temp_distance_bn',"true") == "true"
    nl_temp_external_temp = optional(kwargs,'nl_temp_external_temp',"true") == "true"
    nl_temp_temp_bias = optional(kwargs,'nl_temp_temp_bias',0.1)

    # -- dncnn --
    dcnn_bn = optional(cfg,"dncnn_bn","true") == True
    dcnn_depth = optional(cfg,"dncnn_depth",6)
    dcnn_kernel = optional(cfg,"dncnn_kernel",3)
    dcnn_features = optional(cfg,"dncnn_features",64)

    # -- embedding --
    embedcnn_features = optional(cfg,"embedcnn_features",64)
    embedcnn_depth = optional(cfg,"embedcnn_depth",3)
    embedcnn_kernel = optional(cfg,"embedcnn_kernel",3)
    embedcnn_nplanes_out = optional(cfg,"embedcnn_nplanes_out",8)
    embedcnn_bn = optional(cfg,"embedcnn_bn","true") == "true"
    embedcnn_depth = optional(cfg,"embedcnn_depth",3)

    # -- relevant configs --
    fwd_mode = optional(kwargs,'fwd_mode',"stnls_k")
    ws = optional(kwargs,'ws',-1)
    wt = optional(kwargs,'wt',0)
    k = optional(kwargs,'k',-1)
    sb = optional(kwargs,'sb',None)

    # -- args --
    ninchannels=1
    noutchannels=1
    nl_temp_opt = dict(
        avgpool = nl_temp_avgpool,
        distance_bn = nl_temp_distance_bn,
        external_temp = nl_temp_external_temp,
        temp_bias = nl_temp_temp_bias)
    embedcnn_opt = dict(
        features    = embedcnn_features
        depth       = embedcnn_depth
        kernel      = embedcnn_kernel
        nplanes_out = embedcnn_nplanes_out
        bn          = embedcnn_bn
        depth)      = embedcnn_depth)
    n3block_opt = dict(
        k=nl_k,
        patchsize=nl_patchsize,
        stride=nl_stride,
        temp_opt=nl_temp_opt,
        embedcnn_opt=embedcnn_opt)
    dcnn_opt = dict(
        bn = dcnn_bn,
        depth = dcnn_depth,
        features = dcnn_depth,
        kernel = dcnn_kernel,
        residual = True)

    # -- init model --
    model = N3Net(ninchannels, noutchannels, nfeatures_interm,
                  nblocks=ndncnn, block_opt=dncnn_opt,
                  nl_opt=n3block_opt, residual=False)
    model = model.to(device)

    # -- load weights --
    fdir = Path(__file__).absolute().parents[0] / "../../../" # parent of "./lib"
    state_fn = get_model_weights(fdir,data_sigma,ntype)
    assert os.path.isfile(str(state_fn))

    # -- fill weights --
    load_checkpoint(model,state_fn)

    # -- eval mode as default --
    model.eval()

    return model


