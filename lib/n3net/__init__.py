# -- api --
from . import src_denoising
from . import original
from . import refactored
from . import augmented
from . import shared_model
from . import configs
from . import explore_configs
from . import lightning
from . import utils
from . import flow
from .utils.misc import optional
from .augmented import extract_model_io
from .augmented import extract_model_io as extract_model_config
from .shared_model import ops

# -- api for searching --
from . import search
from .search import get_search,extract_search_config
from .agg import get_agg,extract_agg_config

# -- api for aggregating --
from .shared_model.ops import indexed_matmul_2_efficient
from .augmented import vid_to_raster_inds

# -- papers --
from . import aaai23

def load_model(cfg):
    return get_deno_model(cfg)

def get_deno_model(cfg):
    model_name = optional(cfg,"model_type","augmented")
    device = optional(cfg,"device","cuda:0")
    if model_name == "original":
        model = original.load_model(cfg).to(device)
        return model
    elif model_name == "refactored":
        model = refactored.load_model(cfg).to(device)
        return model
    elif model_name == "augmented":
        model = augmented.load_model(cfg).to(device)
        return model
    else:
        raise ValueError(f"Uknown model [{model_name}]")

def get_model(model_name,mtype,sigma,device,**cfg):
    if model_name == "original":
        model = original.load_model(mtype,sigma).to(device)
        return model
    elif model_name == "refactored":
        model = refactored.load_model(cfg).to(device)
        return model
    else:
        raise ValueError(f"Uknown model [{model_name}]")

