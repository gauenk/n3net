
# -- misc --
import os,math,tqdm
import pprint
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
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

# -- optical flow --
import svnlb

# -- caching results --
import cache_io

# -- network --
import n3net
from torch.autograd import Variable

# -- lightning module --
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint

def run_exp(cfg):

    # -- init results --
    results = edict()
    results.psnrs = []
    results.ssims = []
    results.deno_fn = []
    results.names = []

    # -- data --
    data,loaders = data_hub.sets.load(cfg)
    loader = iter(loaders.te)

    # -- network --
    model = n3net.get_deno_model(cfg.model_name,cfg.sigma,cfg.device)
    model.eval()

    # -- for each batch --
    for batch in tqdm.tqdm(loader):

        # -- unpack --
        noisy,clean = batch['noisy'],batch['clean']
        name = data.te.groups[int(batch['index'][0])]

        # -- select color channel --
        if cfg.bw is True:
            noisy = noisy[:,[0]].contiguous()
            clean = clean[:,[0]].contiguous()

        # -- onto cuda --
        noisy,clean = noisy.to(cfg.device),clean.to(cfg.device)

        # -- normalize images --
        sigma = cfg.sigma/255.
        noisy /= 255.
        clean /= 255.

        # -- size --
        nframes = noisy.shape[0]
        ngroups = int(25 * 37./nframes)
        batch_size = ngroups*1024

        # -- optical flow --
        noisy_np = noisy.cpu().numpy()
        if cfg.comp_flow == "true":
            flows = svnlb.compute_flow(noisy_np,cfg.sigma)
            flows = edict({k:v.to(device) for k,v in flows.items()})
        else:
            flows = None

        # -- internal adaptation --
        run_internal_adapt = cfg.internal_adapt_nsteps > 0
        run_internal_adapt = run_internal_adapt and (cfg.internal_adapt_nepochs > 0)
        if run_internal_adapt:
            model.run_internal_adapt(noisy,cfg.sigma,flows=flows,
                                     ws=cfg.ws,wt=cfg.wt,batch_size=batch_size,
                                     nsteps=cfg.internal_adapt_nsteps,
                                     nepochs=cfg.internal_adapt_nepochs)
        # -- denoise --
        with th.no_grad():
            deno = model(noisy)
            deno = deno.clamp(0.0, 1.0)
            deno = deno.detach()

        # -- save example --
        out_dir = Path(cfg.saved_dir) / get_dir_name(cfg)
        if not out_dir.exists(): out_dir.mkdir(parents=True)
        deno_fn = out_dir / ("deno_%s.png" % name)
        n3net.utils.io.save_image(deno[0],deno_fn)
        # clean_fn = out_dir / ("clean_%s.png" % name)
        # n3net.utils.io.save_image(clean[0],clean_fn)
        # noisy_fn = out_dir / ("noisy_%s.png" % name)
        # n3net.utils.io.save_image(noisy[0],noisy_fn)

        # -- psnr --
        # deno = (deno*255.).type(th.uint8)/255. # convert
        noisy_psnr = n3net.utils.metrics.compute_psnrs(noisy,clean,div=1.)[0]
        psnr = n3net.utils.metrics.compute_psnrs(deno,clean,div=1.)[0]
        ssim = n3net.utils.metrics.compute_ssims(deno,clean,div=1.)[0]
        # print(noisy_psnr)
        # print(psnr,ssim)

        # -- append results --
        results.psnrs.append(psnr)
        results.ssims.append(ssim)
        results.deno_fn.append([deno_fn])
        results.names.append([name])

    return results

def get_dir_name(cfg):
    return "%s/%s/%s_%s" % (cfg.tag,cfg.model_name,cfg.dname,cfg.sigma)

def rescale(img):
    if th.is_tensor(img):
        return th.clamp((img * 255.),0.,255.).type(th.uint8)
    else:
        return np.clip((img * 255.),0.,255.).astype(np.uint8)

def default_cfg():
    # -- config --
    cfg = edict()
    cfg.saved_dir = "./output/saved_results/"
    cfg.checkpoint_dir = "/home/gauenk/Documents/packages/n3net/output/checkpoints/"
    cfg.isize = None
    cfg.num_workers = 4
    cfg.device = "cuda:0"
    cfg.bw = True
    return cfg

def main():

    # -- (0) start info --
    verbose = True
    pid = os.getpid()
    print("PID: ",pid)

    # -- get cache --
    cache_dir = ".cache_io"
    cache_name = "test_rgb_img" # current!
    cache = cache_io.ExpCache(cache_dir,cache_name)
    cache.clear()

    # -- get mesh --
    model_names = ["original"]#,"refactored"]
    dnames = ["urban100"]
    sigmas = [25,50,70]
    internal_adapt_nsteps = [0]
    internal_adapt_nepochs = [5]
    ws,wt = [29],[0]
    comp_flow = ["false"]
    exp_lists = {"sigma":sigmas,"dname":dnames,"model_name":model_names,
                 "internal_adapt_nsteps":internal_adapt_nsteps,
                 "internal_adapt_nepochs":internal_adapt_nepochs,
                 "comp_flow":comp_flow,"ws":ws,"wt":wt}
    exps = cache_io.mesh_pydicts(exp_lists) # create mesh

    # -- group with default --
    cfg = default_cfg()
    cfg.tag = "original"
    cache_io.append_configs(exps,cfg) # merge the two

    # -- run exps --
    nexps = len(exps)
    for exp_num,exp in enumerate(exps):

        # -- info --
        if verbose:
            print("-="*25+"-")
            print(f"Running experiment number {exp_num+1}/{nexps}")
            print("-="*25+"-")
            pp.pprint(exp)

        # -- logic --
        uuid = cache.get_uuid(exp) # assing ID to each Dict in Meshgrid
        # if exp.dname in ["set12","urban100"]: cache.clear_exp(uuid)
        results = cache.load_exp(exp) # possibly load result
        if results is None: # check if no result
            exp.uuid = uuid
            results = run_exp(exp)
            cache.save_exp(uuid,exp,results) # save to cache

    # -- load results --
    records = cache.load_flat_records(exps)

    # -- print by dname,sigma --
    for dname,ddf in records.groupby("dname"):
        field = "model_name"
        for mname,adf in ddf.groupby(field):
            print("mname: ",mname)
            for sigma,sdf in adf.groupby("sigma"):
                ave_psnr = np.stack(sdf.psnrs).mean()
                ave_ssim = np.stack(sdf.ssims).mean()
                print("[%d]: %2.3f,%2.3f" % (sigma,ave_psnr,ave_ssim))


if __name__ == "__main__":
    main()
