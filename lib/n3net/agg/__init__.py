"""

Interface to compare search methods

"""

# -- impl objs --
from .n3agg import N3Agg
from .nlagg import NLAgg

# -- extract config --
from functools import partial
from dev_basics.common import optional as _optional
from dev_basics.common import optional_fields,extract_config,extract_pairs
_fields = []
optional_full = partial(optional_fields,_fields)
extract_agg_config = partial(extract_config,_fields)

def get_agg(cfg):

    # -- unpack --
    init = _optional(cfg,'__init',False) # purposefully weird key
    optional = partial(optional_full,init)
    name = optional(cfg,'name',"n3agg")
    ps = optional(cfg,'ps',10)
    stride0 = optional(cfg,'stride0',5)
    if init: return

    # -- init --
    if name in ["n3agg","original"]:
        return N3Agg(ps,stride0)
    elif name in ["nlagg","ours"]:
        return NLAgg(ps)
    else:
        raise ValueError(f"Uknown search method [{name}]")

# -- fill fields --
get_agg({"__init":True})

