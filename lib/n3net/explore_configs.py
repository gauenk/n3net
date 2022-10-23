"""

Conigs used for exploration!

"""

# -- local --
from . import configs
import cache_io

def base_config():
    # -- get defaults --
    cfg = configs.default_test_vid_cfg()
    cfg.bw = True
    cfg.device = "cuda:0"
    cfg.nframes = 10
    cfg.frame_start = 0
    cfg.frame_end = cfg.frame_start+cfg.nframes-1
    cfg.frame_end = 0 if cfg.frame_end < 0 else cfg.frame_end
    cfg.dname = "set8"
    cfg.sigma = 30.
    cfg.internal_adapt_nsteps = 300
    cfg.internal_adapt_nepochs = 0
    cfg.stride = 5
    cfg.use_train = "false"
    cfg.adapt_mtypes = "rand"
    cfg.model_name = "refactored"
    cfg.flow = "true"
    cfg.vid_name = "sunflower"
    cfg.k = 7
    return cfg

def explore_intro_plot_cfg():
    cfg = base_config()
    ws,wt = [20],[3]
    bs = [1024]
    isizes = ["156_156"]
    exp_lists = {"ws":ws,"wt":wt,"bs":bs,"isize":isizes}
    exps = cache_io.mesh_pydicts(exp_lists) # create mesh
    cfg.nframes = 5
    cfg.frame_end = cfg.frame_start + cfg.nframes - 1
    cache_io.append_configs(exps,cfg)
    return exps

def explore_modulation_cfg():

    #
    # -- [(mem v.s. runtime) -- along batch_size -- ] --
    #

    # -- base --
    cfg = base_config()
    ws,wt = [20],[3]
    bs = [32,1024,10*1024]
    isizes = ["128_128"]
    exp_lists = {"ws":ws,"wt":wt,"bs":bs,"isize":isizes}
    exps = cache_io.mesh_pydicts(exp_lists) # create mesh
    cfg.nframes = 3
    cfg.frame_end = cfg.frame_start + cfg.nframes - 1
    cache_io.append_configs(exps,cfg)
    return exps

def explore_search_space_cfg():

    #
    # -- config grid [(ws,wt) vs (runtime, gpu, psnr)] --
    #

    # -- base --
    cfg = base_config()

    # -- our method's grid --
    bs = [24*1024]
    ws = [15,20,25,30]
    wt = [5,0,1,2,3,4]
    isize = ["128_128"]
    exp_lists = {"ws":ws,"wt":wt,"bs":bs,"isize":isize}
    exps_0 = cache_io.mesh_pydicts(exp_lists) # create mesh
    cache_io.append_configs(exps_0,cfg)

    # -- original method for reference  --
    bs,ws,wt = [-1],[-1],[-1]
    exp_lists = {"ws":ws,"wt":wt,"bs":bs,"isize":isize}
    exps_1 = cache_io.mesh_pydicts(exp_lists) # create mesh
    cfg.flow = "false"
    cfg.model_name = "original"
    cache_io.append_configs(exps_1,cfg)

    # -- combine --
    exps = exps_0# + exps_1

    return exps

def explore_resolution_cfg():
    #
    # -- config grid [(resolution,batch_size) vs (mem,runtime)] --
    #
    ws = [20]
    wt = [3]
    isize = ["512_512","384_384","256_256","128_128","64_64"]
    bs = [512*512*3,384*384*3,256*256*3,128*128*3,64*64*3]
    exp_lists = {"ws":ws,"wt":wt,"bs":bs,"isize":isize}
    exps = cache_io.mesh_pydicts(exp_lists) # create mesh

    cfg = base_config()
    cfg.nframes = 3
    cfg.frame_end = cfg.frame_start + cfg.nframes - 1

    cache_io.append_configs(exps,cfg)

    return exps

