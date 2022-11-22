
# -- misc --
import os,math,tqdm
import pprint,random
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
from n3net import flow

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

def load_trained_state(model,use_train):
    pass

def run_exp(cfg):

    # -- set device --
    th.cuda.set_device(int(cfg.device.split(":")[1]))

    # -- set seed --
    configs.set_seed(cfg.seed)

    # -- init results --
    results = edict()
    results.psnrs = []
    results.ssims = []
    results.noisy_psnrs = []
    results.deno_fns = []
    results.vid_frames = []
    results.vid_name = []
    results.timer_flow = []
    results.timer_deno = []
    results.mem_res = []
    results.mem_alloc = []
    results.flops = []

    # -- network --
    model = n3net.get_deno_model(cfg.model_name,cfg.sigma,cfg)
    model.eval()
    imax = 255.

    # -- optional load trained weights --
    load_trained_state(model,cfg.use_train)

    # -- data --
    data,loaders = data_hub.sets.load(cfg)
    groups = data.te.groups
    indices = [i for i,g in enumerate(groups) if cfg.vid_name in g]

    # -- optional filter --
    frame_start = optional(cfg,"frame_start",0)
    frame_end = optional(cfg,"frame_end",0)
    if frame_start >= 0 and frame_end > 0:
        def fbnds(fnums,lb,ub): return (lb <= np.min(fnums)) and (ub >= np.max(fnums))
        indices = [i for i in indices if fbnds(data.te.paths['fnums'][groups[i]],
                                               cfg.frame_start,cfg.frame_end)]
    for index in indices:

        # -- clean memory --
        th.cuda.empty_cache()
        print("index: ",index)

        # -- unpack --
        sample = data.te[index]
        region = sample['region']
        noisy,clean = sample['noisy'],sample['clean']
        noisy,clean = noisy.to(cfg.device),clean.to(cfg.device)
        vid_frames = sample['fnums'].cpu().numpy()
        print("[%d] noisy.shape: " % index,noisy.shape)

        # -- optional crop --
        noisy = rslice(noisy,region)
        clean = rslice(clean,region)
        print("[%d] noisy.shape: " % index,noisy.shape)

        # -- create timer --
        timer = n3net.utils.timer.ExpTimer()

        # -- size --
        nframes = noisy.shape[0]
        ngroups = int(25 * 37./nframes)
        batch_size = 390*39#ngroups*1024

        # -- optical flow --
        timer.start("flow")
        if cfg.flow == "true":
            sigma_est = flow.est_sigma(noisy)
            print("sigma_est: ",sigma_est)
            flows = flow.run_batch(noisy[None,:],sigma_est)
            # noisy_np = noisy.cpu().numpy()
            # if noisy_np.shape[1] == 1:
            #     noisy_np = np.repeat(noisy_np,3,axis=1)
            # flows = svnlb.compute_flow(noisy_np,cfg.sigma)
            # flows=edict({k:th.from_numpy(v).to(cfg.device) for k,v in flows.items()})
        else:
            flows = None
        timer.stop("flow")

        # -- denoise --
        tsize = 10
        timer.start("deno")
        gpu_mem.print_peak_gpu_stats(False,"val",reset=True)
        with th.no_grad():
            if nframes > tsize:
                deno = temporal_chop(noisy/imax,tsize,model,flows=flows)
            else:
                deno = model(noisy/imax,flows)
        flops = model.flops(noisy)
        deno = th.clamp(deno,0,1.)*imax
        timer.stop("deno")
        mem_alloc,mem_res = gpu_mem.print_peak_gpu_stats(True,"val",reset=True)
        deno = deno.clamp(0.,imax)
        print("deno.shape: ",deno.shape)

        # -- save example --
        out_dir = Path(cfg.saved_dir) / str(cfg.uuid)
        deno_fns = n3net.utils.io.save_burst(deno,out_dir,"deno")

        # -- psnr --
        noisy_psnrs = n3net.utils.metrics.compute_psnrs(noisy,clean,div=imax)
        psnrs = n3net.utils.metrics.compute_psnrs(deno,clean,div=imax)
        ssims = n3net.utils.metrics.compute_ssims(deno,clean,div=imax)
        print(noisy_psnrs)
        print(psnrs)

        # -- append results --
        results.psnrs.append(psnrs)
        results.ssims.append(ssims)
        results.noisy_psnrs.append(noisy_psnrs)
        results.deno_fns.append(deno_fns)
        results.vid_frames.append(vid_frames)
        results.vid_name.append([cfg.vid_name])
        results.mem_res.append([mem_res])
        results.mem_alloc.append([mem_alloc])
        results.flops.append([flops])
        for name,time in timer.items():
            results[name].append(time)

    return results


def main():

    # -- (0) start info --
    verbose = True
    pid = os.getpid()
    print("PID: ",pid)

    # -- get cache --
    cache_dir = ".cache_io"
    cache_name = "explore_grid" # current!
    cache = cache_io.ExpCache(cache_dir,cache_name)
    # cache.clear()

    # -- select configs --
    # exps = explore_configs.explore_modulation_cfg()
    exps = explore_configs.explore_search_space_cfg()
    # exps = expore_configs.explore_resolution_cfg()

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
        # cache.clear_exp(uuid)
        # if exp.model_name == "refactored":
        #     cache.clear_exp(uuid)
        results = cache.load_exp(exp) # possibly load result
        if results is None: # check if no result
            exp.uuid = uuid
            results = run_exp(exp)
            cache.save_exp(uuid,exp,results) # save to cache

    # -- load results --
    records = cache.load_flat_records(exps)
    print(len(records))
    # exit(0)
    # print(records)
    # print(records.filter(like="timer"))

    # -- viz report --
    for use_train,tdf in records.groupby("use_train"):
        for sigma,sdf in tdf.groupby("sigma"):
            print("----- %d -----" % sigma)
            for ca_group,gdf in sdf.groupby("model_name"):
                for use_flow,fdf in gdf.groupby("flow"):
                    agg_psnrs,agg_ssims,agg_dtime = [],[],[]
                    agg_mem_res,agg_mem_alloc = [],[]
                    print("--- %s (%s,%s) ---" % (ca_group,use_train,use_flow))
                    for vname,vdf in fdf.groupby("vid_name"):
                        psnrs = np.stack(vdf['psnrs'])
                        dtime = np.stack(vdf['timer_deno'])
                        mem_res = np.stack(vdf['mem_res'])
                        mem_alloc = np.stack(vdf['mem_alloc'])
                        ssims = np.stack(vdf['ssims'])
                        psnr_mean = psnrs.mean().item()
                        ssim_mean = ssims.mean().item()
                        uuid = vdf['uuid'].iloc[0]

                        # print(dtime,mem_gb)
                        # print(vname,psnr_mean,ssim_mean,uuid)
                        print("%13s: %2.3f %1.3f %s" % (vname,psnr_mean,ssim_mean,uuid))
                        agg_psnrs.append(psnr_mean)
                        agg_ssims.append(ssim_mean)
                        agg_mem_res.append(mem_res.mean().item())
                        agg_mem_alloc.append(mem_alloc.mean().item())
                        agg_dtime.append(dtime.mean().item())
                    psnr_mean = np.mean(agg_psnrs)
                    ssim_mean = np.mean(agg_ssims)
                    dtime_mean = np.mean(agg_dtime)
                    mem_res_mean = np.mean(agg_mem_res)
                    mem_alloc_mean = np.mean(agg_mem_alloc)
                    uuid = tdf['uuid']
                    params = ("Ave",psnr_mean,ssim_mean,dtime_mean,
                              mem_res_mean,mem_alloc_mean)
                    print("%13s: %2.3f %1.3f %2.3f %2.3f %2.3f" % params)


if __name__ == "__main__":
    main()
