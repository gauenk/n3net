from . import src_denoising
from . import original
from . import refactored
from . import shared_model
from . import configs
from . import lightning


def get_deno_model(model_name,sigma,device):
    if model_name == "original":
        model = original.load_model("denoising",sigma).to(device)
        return model
    elif model_name == "refactored":
        model = refactored.load_model("denoising",sigma).to(device)
        return model
    else:
        raise ValueError(f"Uknown model [{model_name}]")

def get_model(model_name,mtype,sigma,device):
    if model_name == "original":
        model = original.load_model(mtype,sigma).to(device)
        return model
    elif model_name == "refactored":
        model = refactored.load_model(mtype,sigma).to(device)
        return model
    else:
        raise ValueError(f"Uknown model [{model_name}]")
