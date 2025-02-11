"""

Test versions of N3net to differences in output due to code modifications.

"""

# -- misc --
import sys,tqdm,pytest,math,random
from pathlib import Path

# -- dict data --
import copy
from easydict import EasyDict as edict

# -- vision --
from PIL import Image

# -- testing --
import unittest
import tempfile

# -- linalg --
import torch as th
import numpy as np
from einops import rearrange,repeat

# -- data --
import data_hub

# -- package imports [to test] --
import stnls # supporting
from torchvision.transforms.functional import center_crop

# -- package imports [to test] --
import n3net
from n3net.utils.gpu_mem import print_gpu_stats,print_peak_gpu_stats
from n3net.utils.misc import rslice_pair

# -- check if reordered --
from scipy import optimize
MAX_NFRAMES = 85
DATA_DIR = Path("./data/")
SAVE_DIR = Path("./output/tests/test_denose_rgb/")
if not SAVE_DIR.exists():
    SAVE_DIR.mkdir(parents=True)

def set_seed(seed):
    random.seed(seed)
    th.manual_seed(seed)
    np.random.seed(seed)
    # th.use_deterministic_algorithms(True)

def pytest_generate_tests(metafunc):
    seed = 123
    set_seed(seed)
    # test_lists = {"ps":[3],"stride":[2],"dilation":[2],
    #               "top":[3],"btm":[57],"left":[7],"right":[57]}
    # test_lists = {"sigma":[50.],"ref_version":["ref","original"]}
    test_lists = {"sigma":[50.],"ref_version":["ref"]}
    for key,val in test_lists.items():
        if key in metafunc.fixturenames:
            metafunc.parametrize(key,val)

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#
# -->  Test original vs refactored code base  <--
#
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def test_original_refactored(sigma,ref_version):

    # -- params --
    device = "cuda:0"
    # vid_set = "sidd_rgb"
    # vid_name = "00"
    # dset = "val"
    vid_set = "set8"
    vid_name = "motorbike"
    isize = "128_128"
    dset = "te"
    flow = False
    noise_version = "blur"
    verbose = True

    # -- setup cfg --
    cfg = edict()
    cfg.dname = vid_set
    cfg.vid_name = vid_name
    cfg.isize = isize
    cfg.sigma = 30.
    cfg.bw = True
    cfg.nframes = 1

    # -- video --
    data,loaders = data_hub.sets.load(cfg)
    groups = data[dset].groups
    indices = [i for i,g in enumerate(groups) if cfg.vid_name in g]
    index = indices[0]

    # -- unpack --
    sample = data[dset][index]
    region = sample['region']
    noisy,clean = sample['noisy'],sample['clean']
    noisy,clean = rslice_pair(noisy,clean,region)
    noisy,clean = noisy.to(device),clean.to(device)
    vid_frames = sample['fnums']
    noisy /= 255.

    # -- flows --
    t,c,h,w = noisy.shape
    flows = edict()
    flows.fflow = th.zeros((t,2,h,w),device=noisy.device)
    flows.bflow = th.zeros((t,2,h,w),device=noisy.device)

    # -- original exec --
    og_model = n3net.original.load_model("denoising",sigma)
    with th.no_grad():
        deno_og = og_model(noisy.clone()).detach()

    # -- refactored exec --
    t,c,h,w = noisy.shape
    region = None#[0,t,0,0,h,w] if ref_version == "ref" else None
    ref_model = n3net.refactored.load_model("denoising",sigma)
    with th.no_grad():
        deno_ref = ref_model(noisy,flows=flows).detach()

    # -- viz --
    if verbose:
        print(deno_og[0,0,:3,:3])
        print(deno_ref[0,0,:3,:3])

    # -- test --
    error = th.mean((deno_og - deno_ref)**2).item()
    if verbose: print("error: ",error)
    assert error < 1e-13

