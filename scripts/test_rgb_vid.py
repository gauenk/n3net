
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
from n3net import lightning
from n3net.utils.misc import optional
import n3net.utils.gpu_mem as gpu_mem
from n3net.utils.misc import rslice,write_pickle,read_pickle
from n3net.utils.proc_utils import spatial_chop,temporal_chop

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
    results.adapt_psnrs = []
    results.deno_fns = []
    results.vid_frames = []
    results.vid_name = []
    results.timer_flow = []
    results.timer_adapt = []
    results.timer_deno = []
    results.mem_res = []
    results.mem_alloc = []

    # -- network --
    model_cfg = n3net.extract_model_io(cfg)
    model = n3net.get_deno_model(**model_cfg)
    model.eval()
    imax = 255.

    # -- optional load trained weights --
    load_trained_state(model,cfg.model_name,cfg.sigma,cfg.use_train)

    # -- data --
    data,loaders = data_hub.sets.load(cfg)
    frame_start = optional(cfg,"frame_start",0)
    frame_end = optional(cfg,"frame_end",-1)
    indices = data_hub.filter_subseq(data[cfg.dset],cfg.vid_name,frame_start,frame_end)

    # # -- optional filter --
    # groups = data.te.groups
    # indices = [i for i,g in enumerate(groups) if cfg.vid_name in g]
    # if frame_start >= 0 and frame_end > 0:
    #     def fbnds(fnums,lb,ub): return (lb <= np.min(fnums)) and (ub >= np.max(fnums))
    #     indices = [i for i in indices if fbnds(data.te.paths['fnums'][groups[i]],
    #                                            cfg.frame_start,cfg.frame_end)]

    for index in indices:

        # -- clean memory --
        th.cuda.empty_cache()
        print("index: ",index)

        # -- unpack --
        sample = data[cfg.dset][index]
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
            flows = flow.run_batch(noisy[None,:],sigma_est)
        else:
            flows = flow.run_zeros(noisy[None,:])
        timer.stop("flow")

        # -- internal adaptation --
        timer.start("adapt")
        run_internal_adapt = cfg.internal_adapt_nsteps > 0
        run_internal_adapt = run_internal_adapt and (cfg.internal_adapt_nepochs > 0)
        adapt_psnrs = [0.]
        if run_internal_adapt:
            adapt_psnrs = model.run_internal_adapt(
                noisy,cfg.sigma,flows=flows,
                ws=cfg.ws,wt=cfg.wt,batch_size=batch_size,
                nsteps=cfg.internal_adapt_nsteps,
                nepochs=cfg.internal_adapt_nepochs,
                sample_mtype=cfg.adapt_mtype,
                clean_gt = clean,
                region_gt = None
            )
        timer.stop("adapt")

        # -- denoise --
        fwd_fxn = get_fwd_fxn(cfg,model)
        # tsize = 10
        timer.start("deno")
        gpu_mem.print_peak_gpu_stats(False,"val",reset=True)
        with th.no_grad():
            # deno = model(noisy/imax,flows)
            deno = fwd_fxn(noisy/imax,flows)
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
        print(psnrs,psnrs.mean())

        # -- append results --
        results.psnrs.append(psnrs)
        results.ssims.append(ssims)
        results.noisy_psnrs.append(noisy_psnrs)
        results.adapt_psnrs.append(adapt_psnrs)
        results.deno_fns.append(deno_fns)
        results.vid_frames.append(vid_frames)
        results.vid_name.append([cfg.vid_name])
        results.mem_res.append([mem_res])
        results.mem_alloc.append([mem_alloc])
        for name,time in timer.items():
            results[name].append(time)

    return results

def get_fwd_fxn(cfg,model):
    s_verbose = True
    t_verbose = True
    s_size = cfg.spatial_crop_size
    s_overlap = cfg.spatial_crop_overlap
    t_size = cfg.temporal_crop_size
    t_overlap = cfg.temporal_crop_overlap
    model_fwd = lambda vid,flows: model(vid,flows=flows)
    if not(s_size is None) and not(s_size == "none"):
        schop_p = lambda vid,flows: spatial_chop(s_size,s_overlap,model_fwd,vid,
                                                 flows=flows,verbose=s_verbose)
    else:
        schop_p = model_fwd
    tchop_p = lambda vid,flows: temporal_chop(t_size,t_overlap,schop_p,vid,
                                              flows=flows,verbose=t_verbose)
    fwd_fxn = tchop_p # rename
    return fwd_fxn

def load_trained_state(model,name,sigma,use_train):

    # -- skip if needed --
    if not(use_train == "true"): return
    ca_fwd = "dnls_k"

    if ca_fwd == "dnls_k":
        if abs(sigma-50.) < 1e-10:
            model_path = "b118f3a8-f1bf-43b5-9853-b0d346c548a9-epoch=02.ckpt"
            # model_path = "26cc2011-dca1-43ae-aa41-0da32a259274-epoch=58.ckpt"
            # model_path = "26cc2011-dca1-43ae-aa41-0da32a259274-epoch=58.ckpt"
            # model_path = "b955079e-3224-40c8-8e8b-1e719aa0c8d7-epoch=28.ckpt"
            # model_path = "c4a39d49-d006-4015-91e8-feece2625beb-epoch=28.ckpt" # 50.
        else:
            if name == "original":
                # model_path = "98a8f7b0-828b-413c-a61d-208c2accd630-epoch=48.ckpt"
                model_path = "98a8f7b0-828b-413c-a61d-208c2accd630-epoch=04-val_loss=8.97e-04.ckpt"
            else:
                # model_path = "9587c811-0efc-44dc-be5b-5ccfa4eb819c-epoch=41.ckpt"
                # model_path = "70215fee-2f73-4972-98bf-edc547bb31a0-epoch=41.ckpt"
                # model_path = "b118f3a8-f1bf-43b5-9853-b0d346c548a9-epoch=02.ckpt"
                model_path = "12b10a3f-0148-4e39-a5c3-812ae0bab529-epoch=05.ckpt"
                # model_path = "b118f3a8-f1bf-43b5-9853-b0d346c548a9-epoch=02.ckpt"
            # model_path = "9587c811-0efc-44dc-be5b-5ccfa4eb819c-epoch=19.ckpt"
            # model_path = "97cab11c-f0f5-4563-88bb-5051d730931e-epoch=68.ckpt"
            # model_path = "97cab11c-f0f5-4563-88bb-5051d730931e-epoch=52.ckpt"
            # model_path = "8c0c2ca0-28d4-4625-8910-8595a9068970-epoch=04-val_loss=8.36e-04.ckpt"
            # model_path = "8c0c2ca0-28d4-4625-8910-8595a9068970-epoch=44.ckpt"
            # model_path = "d2e0f65a-1fb2-4667-af31-3c3fe8e36ef0-epoch=58.ckpt"
            # model_path = "d2e0f65a-1fb2-4667-af31-3c3fe8e36ef0-epoch=02-val_loss=8.35e-04.ckpt"
            # model_path = "c7797a88-84d5-47e3-a76c-e3f867476583-epoch=28.ckpt"
            # model_path = "59399328-ac47-4c81-a590-78afea4e5342-epoch=08.ckpt" # 25.
        # model_path = "2539a251-8233-49a8-bb4f-db68e8c96559-epoch=99.ckpt"
        # model_path = "2539a251-8233-49a8-bb4f-db68e8c96559-epoch=81-val_loss=1.24e-03.ckpt"
        # model_path = "2539a251-8233-49a8-bb4f-db68e8c96559-epoch=38-val_loss=1.15e-03.ckpt"
        # model_path = "2539a251-8233-49a8-bb4f-db68e8c96559-epoch=26.ckpt"
    elif ca_fwd == "default":
        model_path = "dec78611-36a7-4a9e-8420-4e60fe8ea358-epoch=91-val_loss=6.63e-04.ckpt"
    else:
        raise ValueError(f"Uknown ca_fwd [{ca_fwd}]")
    model_path = str(Path("output/checkpoints/") / model_path)

    # -- load model state --
    print("Loading state: ",model_path)
    state = th.load(model_path)['state_dict']
    lightning.remove_lightning_load_state(state)
    model.load_state_dict(state)
    return model

def main():

    # -- (0) start info --
    verbose = True
    pid = os.getpid()
    print("PID: ",pid)

    # -- get cache --
    cache_dir = ".cache_io"
    # cache_name = "test_rgb_net" # current!
    cache_name = "test_rgb_net_rebuttle_testing" # current!
    # cache_name = "test_rgb_net_rebuttle_s50" # current!
    # cache_name = "test_rgb_net_rebuttle_s50_te" # current!
    # cache_name = "test_rgb_net_rebuttle_s25" # current!
    # cache_name = "test_rgb_net_rebuttle_s25_te" # current!
    cache = cache_io.ExpCache(cache_dir,cache_name)
    # cache.clear()

    # -- get defaults --
    cfg = configs.default_test_vid_cfg()
    cfg.seed = 123
    cfg.bw = True
    cfg.nframes = 6
    cfg.isize = "none"
    # cfg.isize = "512_512"
    cfg.isize = "256_256"
    cfg.cropmode = "center"
    cfg.frame_start = 0
    cfg.frame_end = cfg.frame_start+cfg.nframes-1
    cfg.frame_end = 0 if cfg.frame_end < 0 else cfg.frame_end
    cfg.spatial_crop_size = "none"
    cfg.spatial_crop_overlap = 0.#0.1
    cfg.temporal_crop_size = cfg.nframes
    cfg.temporal_crop_overlap = 0/5.#4/5. # 3 of 5 frames
    cfg.ps = 10

    # -- get mesh --
    internal_adapt_nsteps = [300]
    internal_adapt_nepochs = [0]
    k,bs,stride = [7],[28*1024],[5]
    # ws,wt,k,bs,stride = [20],[0],[7],[28*1024],[5]
    # ws,wt,k,bs,stride = [29],[3],[7],[28*1024],[5]
    # sigmas = [10.,30.]
    # sigmas = [30.,50.]
    sigmas = [25.]
    # sigmas = [30.]
    # sigmas = [50.]
    # ws,wt = [29],[3]
    # sigmas = [50.]
    ws,wt = [21],[3]
    # ws,wt = [15],[0]
    dnames = ["set8"]
    use_train = ["true"]
    # use_train = ["true","false"]
    # sigmas = [50.]
    # ws,wt,k,bs,stride = [15],[3],[7],[32],[5]
    # wt,sigmas = [0],[30.]
    # vid_names = ["tractor"]
    # bs = [512*512]
    # vid_names = ["sunflower"]#,"hypersmooth","tractor"]
    vid_names = ["rafting"]
    # vid_names = ["rafting","sunflow"]
    # vid_names = ["sunflower","tractor","snowboard","motorbike",
    #              "hypersmooth","park_joy","rafting","touchdown"]
    flow = ["true"]
    # flow = ["true","false"]
    model_names = ["refactored","original"]
    # model_names = ["original"]
    # model_names = ["refactored"]
    adapt_mtypes = ["rand"]
    exp_lists = {"dname":dnames,"vid_name":vid_names,"sigma":sigmas,
                 "internal_adapt_nsteps":internal_adapt_nsteps,
                 "internal_adapt_nepochs":internal_adapt_nepochs,
                 "flow":flow,"ws":ws,"wt":wt,"adapt_mtype":adapt_mtypes,
                 "use_train":use_train,"stride":stride,
                 "ws":ws,"wt":wt,"k":k, "bs":bs, "model_name":model_names}
    exps_a = cache_io.mesh_pydicts(exp_lists) # create mesh
    # exp_lists['wt'] = [3]
    # exp_lists['bs'] = [512*512//8]
    # exps_a = cache_io.mesh_pydicts(exp_lists)
    # exp_lists['ws'] = [29]
    # exp_lists['wt'] = [0]
    # exps_a += cache_io.mesh_pydicts(exp_lists) # create mesh
    cache_io.append_configs(exps_a,cfg) # merge the two


    # -- original w/out training --
    # exp_lists['ws'] = [-1]
    # exp_lists['wt'] = [-1]
    # exp_lists['bs'] = [-1]
    exp_lists['model_name'] = ["original"]
    exp_lists['flow'] = ["false"]
    exp_lists['use_train'] = ["false"]#,"true"]
    exps_b = cache_io.mesh_pydicts(exp_lists) # create mesh
    cache_io.append_configs(exps_b,cfg) # merge the two

    # -- cat exps --
    exps = exps_a# + exps_b
    # exps = exps_b

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
        cache.clear_exp(uuid)
        # if exp.model_name != "refactored":
        #     cache.clear_exp(uuid)
        results = cache.load_exp(exp) # possibly load result
        if results is None: # check if no result
            exp.uuid = uuid
            results = run_exp(exp)
            cache.save_exp(uuid,exp,results) # save to cache

    # -- load results --
    records = cache.load_flat_records(exps)
    # print(records[['timer_deno','model_name','mem_res']])
    # exit(0)
    # print(records)
    # print(records.filter(like="timer"))

    # -- viz report --
    for use_train,tdf in records.groupby("use_train"):
        for wt,wdf in tdf.groupby("wt"):
            for sigma,sdf in wdf.groupby("sigma"):
                print("----- %d -----" % sigma)
                for ca_group,gdf in sdf.groupby("model_name"):
                    for use_flow,fdf in gdf.groupby("flow"):
                        agg_psnrs,agg_ssims,agg_dtime = [],[],[]
                        agg_mem_res,agg_mem_alloc = [],[]
                        print("--- %s (%s,%s,%s) ---" % (ca_group,use_train,use_flow,wt))
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
