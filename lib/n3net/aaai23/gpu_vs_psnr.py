
# -- misc --
import os,math,tqdm
import pprint
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

def default_cfg():
    # -- config --
    cfg = edict()
    cfg.saved_dir = "./output/saved_results/"
    cfg.checkpoint_dir = "/home/gauenk/Documents/packages/n3net/output/checkpoints/"
    cfg.num_workers = 1
    cfg.device = "cuda:0"
    cfg.mtype = "gray"
    cfg.seed = 123
    return cfg

def append_detailed_cfg(cfg):
    # -- add to cfg --
    cfg.adapt_mtype = "rand"
    cfg.internal_adapt_nsteps = 300
    cfg.internal_adapt_nepochs = 0
    cfg.ws = 15
    cfg.wt = 3
    cfg.bs = 32#256#64*1024 # 128*1024
    cfg.bw = True
    cfg.k = 7
    cfg.stride = 5
    def_cfg = default_cfg()
    cfg_l = [cfg]
    cache_io.append_configs(cfg_l,def_cfg) # merge the two
    return cfg_l[0]

def load_proposed(cfg,use_train="false",flow="true"):
    mtype = "refactored"
    return load_results(cfg,mtype,flow,use_train)

def load_original(cfg):
    mtype = "original"
    use_train= "false"
    flow = "false"
    return load_results(cfg,mtype,flow,use_train)

def load_results(cfg,mtype,flow,use_train):

    # -- get cache --
    lidia_home = Path(__file__).parents[0] / "../../../"
    cache_dir = str(lidia_home / ".cache_io")
    cache_name = "test_rgb_net" # current!
    cache = cache_io.ExpCache(cache_dir,cache_name)
    cfg = append_detailed_cfg(cfg)

    # -- config --
    cfg.flow = flow
    cfg.model_name = mtype
    cfg.use_train = use_train

    # -- load results --
    cfg_l = [cfg]
    pp.pprint(cfg_l[0])
    records = cache.load_flat_records(cfg_l)
    return records
