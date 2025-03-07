"""

Interface to compare search methods

"""

# -- impl objs --
from .n3nl import N3NLSearch
from .nl import NLSearch

# -- extract config --
from functools import partial
from dev_basics.common import optional as _optional
from dev_basics.common import optional_fields,extract_config,extract_pairs
_fields = []
optional_full = partial(optional_fields,_fields)
extract_search_config = partial(extract_config,_fields)

def get_search(cfg):

    # -- unpack --
    init = _optional(cfg,'__init',False) # purposefully weird key
    optional = partial(optional_full,init)
    name = optional(cfg,'name',"n3nl")
    k = optional(cfg,'k',7)
    ps = optional(cfg,'ps',10)
    nheads = optional(cfg,'nheads',1)
    stride0 = optional(cfg,'stride0',5)
    stride1 = optional(cfg,'stride1',1)
    ws = optional(cfg,'ws',15)
    wt = optional(cfg,'wt',0)
    if init: return

    # -- check --
    assert nheads == 1,"Must be one head"

    # -- init --
    if name in ["n3nl","original"]:
        return N3NLSearch(ps,ws,stride0,stride1)
    elif name in ["nl","ours"]:
        return NLSearch(k,ps,ws,wt,nheads,stride0,stride1)
    else:
        raise ValueError(f"Uknown search method [{name}]")

# -- fill fields --
get_search({"__init":True})

