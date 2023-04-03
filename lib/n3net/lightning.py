

# -- misc --
import os,math,tqdm
import pprint,copy
pp = pprint.PrettyPrinter(indent=4)

# -- linalg --
import numpy as np
import torch as th
from einops import rearrange,repeat

# -- data mngmnt --
from pathlib import Path
from easydict import EasyDict as edict

# -- data --
import stnls
import data_hub

# -- optical flow --
from dev_basics import flow
from dev_basics.utils.misc import rslice,write_pickle,read_pickle
from dev_basics.utils.metrics import compute_psnrs,compute_ssims
from dev_basics.utils.timer import ExpTimer
import dev_basics.utils.gpu_mem as gpu_mem


# -- network --
import n3net
import n3net.configs as configs
import n3net.utils.gpu_mem as gpu_mem
from n3net.utils.timer import ExpTimer
from n3net.utils.metrics import compute_psnrs,compute_ssims
from n3net.utils.misc import rslice,write_pickle,read_pickle
from n3net.utils.model_utils import remove_lightning_load_state

# -- generic logging --
import logging
logging.basicConfig()

# -- lightning module --
import torch
import pytorch_lightning as pl
from pytorch_lightning import Callback
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.distributed import rank_zero_only


class N3NetLit(pl.LightningModule):

    # def __init__(self,model_cfg,batch_size=1,flow=True,
    #              isize=None,bw=False,
    #              lr_init=1e-3,lr_final=1e-8,weight_decay=1e-4,nepochs=0,
    #              warmup_epochs=0,scheduler="default",
    #              task=0,uuid="",sim_type="g",sim_device="cuda:0"):

    def __init__(self,model_cfg,batch_size=1,flow=True,isize=None,bw=False,
                 lr_init=0.0002,lr_final=1e-8,weight_decay=0.,nepochs=30,
                 warmup_epochs=0,scheduler="default",
                 task="denoise",uuid="",sim_type="g",sim_device="cuda:0"):
        super().__init__()
        self.model = n3net.get_deno_model(model_cfg)
        self.model.train()
        self.warmup_epochs = warmup_epochs
        self.sigma = model_cfg.sigma
        self.batch_size = batch_size
        self.flow = flow
        self.isize = isize
        self.lr_init = lr_init
        self.lr_final = lr_final
        self.weight_decay = weight_decay
        self.nepochs = nepochs
        self.scheduler = scheduler
        self.task = task
        self.uuid = uuid
        self.gen_loger = logging.getLogger('lightning')
        self.gen_loger.setLevel("NOTSET")

    def forward(self,vid):
        flows = flow.orun(vid,self.flow)
        deno = self.model(vid,flows=flows)
        return deno

    def configure_optimizers(self):
        optim = th.optim.Adam(self.parameters(),lr=self.lr_init,
                              weight_decay=self.weight_decay)
        sched = self.configure_scheduler(optim)
        return [optim], [sched]

    def configure_scheduler(self,optim):
        # StepLR = th.optim.lr_scheduler.StepLR
        if self.scheduler in ["default","exp_decay"]:
            gamma = 1-math.exp(math.log(self.lr_final/self.lr_init)/self.nepochs)
            ExponentialLR = th.optim.lr_scheduler.ExponentialLR
            scheduler = ExponentialLR(optim,gamma=gamma) # (.995)^50 ~= .78
        else:
            raise ValueError(f"Uknown scheduler [{self.scheduler}]")
        return scheduler

    def training_step(self, batch, batch_idx):

        # -- each sample in batch --
        loss = 0 # init @ zero
        nbatch = len(batch['noisy'])
        denos,cleans = [],[]
        for i in range(nbatch):
            deno_i,clean_i,loss_i = self.training_step_i(batch, i)
            loss += loss_i
            denos.append(deno_i)
            cleans.append(clean_i)
        loss = loss / nbatch

        # -- append --
        denos = th.stack(denos)
        cleans = th.stack(cleans)

        # -- log --
        self.log("train_loss", loss.item(), on_step=True,
                 on_epoch=False,batch_size=self.batch_size)

        # -- terminal log --
        psnr = np.mean(compute_psnrs(denos,cleans,div=1.)).item()
        self.gen_loger.info("train_psnr: %2.2f" % psnr)
        # print("train_psnr: %2.2f" % val_psnr)
        self.log("train_loss", loss.item(), on_step=True,
                 on_epoch=False, batch_size=self.batch_size)
        self.log("train_psnr", psnr, on_step=True,
                 on_epoch=False, batch_size=self.batch_size)

        return loss

    def training_step_i(self, batch, i):

        # -- unpack batch
        noisy = batch['noisy'][i]/255.
        clean = batch['clean'][i]/255.
        # print("[train] noisy.shape: ",noisy.shape)

        # -- foward --
        deno = self.forward(noisy)

        # -- report loss --
        loss = th.mean((clean - deno)**2)
        return deno.detach(),clean,loss

    def validation_step(self, batch, batch_idx):

        # -- denoise --
        noisy,clean = batch['noisy'][0]/255.,batch['clean'][0]/255.
        region = batch['region'][0]
        noisy = rslice(noisy,region)
        clean = rslice(clean,region)
        # print("[val] noisy.shape: ",noisy.shape)

        # -- forward --
        gpu_mem.print_peak_gpu_stats(False,"val",reset=True)
        with th.no_grad():
            deno = self.forward(noisy)
        _,mem_gb = gpu_mem.print_peak_gpu_stats(False,"val",reset=True)

        # -- loss --
        loss = th.mean((clean - deno)**2)

        # -- report --
        self.log("val_loss", loss.item(), on_step=False,
                 on_epoch=True,batch_size=1,sync_dist=True)
        self.log("val_gpu_mem", mem_gb, on_step=False,
                 on_epoch=True,batch_size=1,sync_dist=True)

        # -- terminal log --
        val_psnr = np.mean(compute_psnrs(deno,clean,div=1.)).item()
        self.gen_loger.info("val_psnr: %2.2f" % val_psnr)

    def test_step(self, batch, batch_nb):

        # -- denoise --
        index,region = batch['index'][0],batch['region'][0]
        noisy,clean = batch['noisy'][0]/255.,batch['clean'][0]/255.
        noisy = rslice(noisy,region)
        clean = rslice(clean,region)
        # print("[test] noisy.shape: ",noisy.shape)

        # -- forward --
        gpu_mem.print_peak_gpu_stats(False,"test",reset=True)
        with th.no_grad():
            deno = self.forward(noisy)
        _,mem_gb = gpu_mem.print_peak_gpu_stats(False,"test",reset=True)

        # -- compare --
        loss = th.mean((clean - deno)**2)
        psnr = np.mean(compute_psnrs(deno,clean,div=1.)).item()
        ssim = np.mean(compute_ssims(deno,clean,div=1.)).item()

        # -- terminal log --
        self.log("psnr", psnr, on_step=True, on_epoch=False, batch_size=1)
        self.log("ssim", ssim, on_step=True, on_epoch=False, batch_size=1)
        self.log("index",  int(index.item()),on_step=True,on_epoch=False,batch_size=1)
        self.log("mem_gb",  mem_gb, on_step=True, on_epoch=False, batch_size=1)
        self.gen_loger.info("te_psnr: %2.2f" % psnr)

        # -- log --
        results = edict()
        results.test_loss = loss.item()
        results.test_psnr = psnr
        results.test_ssim = ssim
        results.test_gpu_mem = mem_gb
        results.test_index = index.cpu().numpy().item()
        return results

class MetricsCallback(Callback):
    """PyTorch Lightning metric callback."""

    def __init__(self):
        super().__init__()
        self.metrics = {}

    def _accumulate_results(self,each_me):
        for key,val in each_me.items():
            if not(key in self.metrics):
                self.metrics[key] = []
            if hasattr(val,"ndim"):
                ndim = val.ndim
                val = val.cpu().numpy().item()
            self.metrics[key].append(val)

    @rank_zero_only
    def log_metrics(self, metrics, step):
        # metrics is a dictionary of metric names and values
        # your code to record metrics goes here
        print("logging metrics: ",metrics,step)

    def on_train_epoch_end(self, trainer, pl_module):
        each_me = copy.deepcopy(trainer.callback_metrics)
        self._accumulate_results(each_me)

    def on_validation_epoch_end(self, trainer, pl_module):
        each_me = copy.deepcopy(trainer.callback_metrics)
        self._accumulate_results(each_me)

    def on_test_epoch_end(self, trainer, pl_module):
        each_me = copy.deepcopy(trainer.callback_metrics)
        self._accumulate_results(each_me)

    def on_train_batch_end(self, trainer, pl_module, outs,
                           batch, batch_idx, dl_idx):
        each_me = copy.deepcopy(trainer.callback_metrics)
        self._accumulate_results(each_me)


    def on_validation_batch_end(self, trainer, pl_module, outs,
                                batch, batch_idx, dl_idx):
        each_me = copy.deepcopy(trainer.callback_metrics)
        self._accumulate_results(each_me)

    def on_test_batch_end(self, trainer, pl_module, outs,
                          batch, batch_idx, dl_idx):
        self._accumulate_results(outs)

