import torch as th
import numpy as np
import torch.nn as nn
from pathlib import Path
from collections import OrderedDict

def freeze(model):
    for p in model.parameters():
        p.requires_grad=False

def unfreeze(model):
    for p in model.parameters():
        p.requires_grad=True

def is_frozen(model):
    x = [p.requires_grad for p in model.parameters()]
    return not all(x)

def save_checkpoint(model_dir, state, session):
    epoch = state['epoch']
    substr = "model_epoch_{}_{}.pth".format(epoch,session)
    model_out_path = str(Path(model_dir) / substr)
    th.save(state, model_out_path)

def select_sigma(sigma):
    sigmas = np.array([25, 50, 70])
    msigma = np.argmin((sigmas - sigma)**2)
    return sigmas[msigma]

def remove_lightning_load_state(state):
    names = list(state.keys())
    for name in names:
        name_new = name.split(".")[1:]
        name_new = ".".join(name_new)
        state[name_new] = state[name]
        del state[name]

def load_checkpoint(model, path, wtype="git"):
    if wtype in ["git","original"]:
        load_checkpoint_git(model,path)
    elif wtype in ["lightning","lit"]:
        load_checkpoint_lit(model,path)
    else:
        raise ValueError(f"Uknown checkpoint weight type [{wtype}]")

def load_checkpoint_lit(model,path):
    # -- filename --
    if not Path(path).exists():
        path = str("output/checkpoints/" / Path(path))
    assert Path(path).exists()
    weights = th.load(path)
    state = weights['state_dict']
    remove_lightning_load_state(state)
    model.load_state_dict(state)

def load_checkpoint_git(model,path):
    # -- filename --
    if not Path(path).exists():
        path = str("output/checkpoints/" / Path(path))
    assert Path(path).exists()
    checkpoint = th.load(path)
    try:
        # model.load_state_dict(checkpoint["state_dict"])
        raise ValueError("")
    except Exception as e:
        state_dict = checkpoint["net"]
        # new_state_dict = OrderedDict()
        # for k, v in state_dict.items():
        #     name = k[7:] if 'module.' in k else k
        #     new_state_dict[name] = v
        model.load_state_dict(state_dict)

def load_checkpoint_multigpu(model, weights):
    checkpoint = th.load(weights)
    state_dict = checkpoint["state_dict"]
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)

def load_start_epoch(weights):
    checkpoint = th.load(weights)
    epoch = checkpoint["epoch"]
    return epoch

def load_optim(optimizer, weights):
    checkpoint = th.load(weights)
    optimizer.load_state_dict(checkpoint['optimizer'])
    for p in optimizer.param_groups: lr = p['lr']
    return lr

def get_model_weights(fdir,data_sigma,ntype):
    path = Path(fdir) / "weights"
    if ntype == "gaussian":
        path /= "results_gaussian_denoising"
        model_sigma = select_sigma(data_sigma)
        mdir = "pretrained_sigma%d" % model_sigma
        mdir_full = path / mdir / "checkpoint" / "051_ckpt.t7"
    elif ntype == "poisson":
        path /= "results_poissongaussian_denoising"
        mdir = "pretrained"
        mdir_full = path / mdir / "checkpoint" / "051_ckpt.t7"
    else:
        raise ValueError(f"Uknown noise type [{ntype}]")
    return str(mdir_full)



def temporal_chop(x,tsize,fwd_fxn,flows=None):
    nframes = x.shape[0]
    nslice = (nframes-1)//tsize+1
    x_agg = []
    for ti in range(nslice):
        ts = ti*tsize
        te = min((ti+1)*tsize,nframes)
        tslice = slice(ts,te)
        if flows:
            x_t = fwd_fxn(x[tslice],flows)
        else:
            x_t = fwd_fxn(x[tslice])
        x_agg.append(x_t)
    x_agg = th.cat(x_agg)
    return x_agg

