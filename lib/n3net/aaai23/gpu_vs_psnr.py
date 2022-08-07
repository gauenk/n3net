


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

def load_intro(cfg):

    # -- get cache --
    lidia_home = Path(__file__).parents[0] / "../../../"
    cache_dir = str(lidia_home / ".cache_io")
    cache_name = "test_rgb_net" # current!
    cache = cache_io.ExpCache(cache_dir,cache_name)

    # -- add to cfg --
    cfg.nframes = 10
    cfg.frame_start = 10
    cfg.frame_end = cfg.frame_start + cfg.nframes - 1
    cfg.adapt_mtype = "rand"
    cfg.internal_adapt_nsteps = 300
    cfg.internal_adapt_nepochs = 0
    cfg.ws = 15
    cfg.wt = 3
    cfg.bs = 256#64*1024 # 128*1024
    cfg.bw = True
    cfg.k = 7
    cfg.stride = 5
    cfg.isize = "256_256"
    cfg.model_name = "refactored"
    cfg.use_train = "false"
    cfg.flow = "true"
    def_cfg = default_cfg()
    cfg_l = [cfg]
    cache_io.append_configs(cfg_l,def_cfg) # merge the two

    # -- load results --
    # pp.pprint(cfg_l[0])
    records = cache.load_flat_records(cfg_l)
    return records
