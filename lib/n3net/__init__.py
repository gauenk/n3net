# -- api --
from . import src_denoising
from . import original
from . import refactored
from . import shared_model
from . import configs
from . import explore_configs
from . import lightning
from . import utils
from . import flow
from .utils.misc import optional
from .refactored import extract_model_io

# -- papers --
from . import aaai23

# -- model io --
def get_deno_model(model_name,sigma,**kwargs):
    device = optional(kwargs,"device","cuda:0")
    if model_name == "original":
        model = original.load_model("denoising",sigma,kwargs).to(device)
        return model
    elif model_name == "refactored":
        model = refactored.load_model(**kwargs).to(device)
        return model
    else:
        raise ValueError(f"Uknown model [{model_name}]")

def get_model(model_name,mtype,sigma,device,**kwargs):
    if model_name == "original":
        model = original.load_model(mtype,sigma).to(device)
        return model
    elif model_name == "refactored":
        model = refactored.load_model(**kwargs).to(device)
        return model
    else:
        raise ValueError(f"Uknown model [{model_name}]")

