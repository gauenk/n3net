
# -- our search fxn --
from . import nl_search
from .nl_search import NLSearch

# -- extracting config --
from functools import partial
from ..utils import optional as _optional

# -- auto populate fields to extract config --
_fields = []
def optional_full(init,pydict,field,default):
    if not(field in _fields) and init:
        _fields.append(field)
    return _optional(pydict,field,default)

# -- init search --
def init_search(*args,**kwargs):

    # -- allows for all keys to be aggregated at init --
    init = _optional(kwargs,'__init',False) # purposefully weird key
    optional = partial(optional_full,init)

    # -- relevant configs --
    k = optional(kwargs,'k',-1)
    ps = optional(kwargs,'ps',7)
    nheads = optional(kwargs,'nheads',1)
    stride0 = optional(kwargs,'stride0',1)
    stride1 = optional(kwargs,'stride1',1)
    ws = optional(kwargs,'ws',15)
    wt = optional(kwargs,'wt',0)
    nbwd = optional(kwargs,'nbwd',1)
    rbwd = optional(kwargs,'rbwd',False)
    exact = optional(kwargs,'exact',False)
    bs = optional(kwargs,'bs',-1)
    dil = optional(kwargs,'dilation',1)
    chnls = optional(kwargs,'chnls',-1)
    index_reset = optional(kwargs,'index_reset',-1)
    use_k = optional(kwargs,'use_k',k>0)
    include_self = optional(kwargs,'include_self',False)
    # name = optional(kwargs,'sfxn','prod')

    # -- break here if init --
    if init: return

    # -- init model --
    search = NLSearch(k=k, ps=ps, ws=ws, nheads=nheads,
                      chnls=chnls, dilation=dil,
                      stride0=stride0, stride1=stride1,
                      index_reset=index_reset,include_self=include_self,
                      use_k=use_k)
    return search


# -- run to populate "_fields" --
init_search(__init=True)

def extract_search_config(cfg):
    # -- auto populated fields --
    fields = _fields
    _cfg = {}
    for field in fields:
        if field in cfg:
            _cfg[field] = cfg[field]
    return _cfg
