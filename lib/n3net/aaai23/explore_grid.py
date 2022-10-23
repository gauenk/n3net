# -- misc --
import os,math,tqdm
import pprint
pp = pprint.PrettyPrinter(indent=4)
import copy
dcopy = copy.deepcopy

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
import n3net.configs as configs
import n3net.explore_configs as explore_configs
from n3net import lightning
from n3net.utils.misc import optional
import n3net.utils.gpu_mem as gpu_mem
from n3net.utils.misc import rslice,write_pickle,read_pickle
from n3net.utils.model_utils import temporal_chop

def load_results(cfg):

    # -- load cache --
    n3net_home = Path(__file__).parents[0] / "../../../"
    cache_dir = str(n3net_home / ".cache_io")
    cache_name = "explore_grid" # current!
    cache = cache_io.ExpCache(cache_dir,cache_name)

    # -- get defaults --
    # cfg = configs.default_test_vid_cfg()
    # cfg.bw = True
    # cfg.device = "cuda:0"
    # cfg.nframes = 10
    # cfg.frame_start = 0
    # cfg.frame_end = cfg.frame_start+cfg.nframes-1
    # cfg.frame_end = 0 if cfg.frame_end < 0 else cfg.frame_end
    # cfg.dname = "set8"
    # cfg.sigma = 30.
    # cfg.internal_adapt_nsteps = 300
    # cfg.internal_adapt_nepochs = 0
    # cfg.stride = 5
    # cfg.use_train = "false"
    # cfg.adapt_mtypes = "rand"
    # cfg.model_name = "refactored"
    # cfg.flow = "true"

    #
    # -- select configs --
    #
    # exps = explore_configs.explore_modulation_cfg()
    exps = explore_configs.explore_search_space_cfg()
    # exps = expore_configs.explore_resolution_cfg()

    #
    # -- TO REMOVE --
    #

    # # -- exps [1/3] --
    # vid_names = ["sunflower"]#,"hypersmooth","tractor"]
    # ws,wt,k = [20],[3],[7]
    # bs = [32,1024,10*1024]
    # isizes = ["128_128"]
    # exp_lists = {"vid_name":vid_names,"ws":ws,"wt":wt,
    #              "k":k,"bs":bs,"isize":isizes}
    # exps_a = cache_io.mesh_pydicts(exp_lists) # create mesh
    # cache_io.append_configs(exps_a,cfg)

    # # -- exps [2/3] --
    # exp_lists['ws'] = [20]
    # exp_lists['wt'] = [3]
    # exp_lists['isize'] = ['96_96','128_128','156_156']
    # exps_b = cache_io.mesh_pydicts(exp_lists) # create mesh
    # cache_io.append_configs(exps_b,cfg)

    # # -- exps [3/3] --
    # exp_lists['ws'] = [20]
    # exp_lists['wt'] = [3]
    # exp_lists['isize'] = ["512_512","384_384","256_256","128_128","64_64"]
    # exp_lists['bs'] = [512*512*3,384*384*3,256*256*3,128*128*3,64*64*3]
    # # exp_lists['isize'] = ["260_260","220_220","180_180","140_140","100_100","60_60"]
    # # exp_lists['bs'] = [260*260*3,220*220*3,180*180*3,140*140*3,100*100*3,60*60*3]
    # exps_c = cache_io.mesh_pydicts(exp_lists) # create mesh
    # cache_io.append_configs(exps_c,cfg)

    # # -- combine --
    # exps = exps_c # exps_a + exps_b
    # cfg.nframes = 3
    # cfg.frame_end = cfg.frame_start + cfg.nframes - 1
    # cache_io.append_configs(exps,cfg) # merge the two

    # for i in range(len(exps)):
    #     pp.pprint(exps[i])
    # exit(0)

    pp.pprint(exps[0])

    # -- read --
    root = Path("./.aaai23")
    if not root.exists():
        root.mkdir()
    pickle_store = str(root / "n3net_explore_grid.pkl")
    records = cache.load_flat_records(exps,save_agg=pickle_store,clear=False)

    # -- standardize col names --
    records = records.rename(columns={"bs":"batch_size"})
    # print(records[['ws','wt','isize','batch_size']])
    # print(len(records))
    assert len(records) > 0

    return records
