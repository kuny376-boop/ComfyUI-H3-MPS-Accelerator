"""Independent Spark-inspired reference, not an official Spark-H3 port.
Inputs [B,H,N,D], inference only. Query-block streaming bounds score memory.
"""
import math
import torch

@torch.inference_mode()
def sparse_reference(q, k, v, density=0.2, block=64, weighted=True, protected=0):
    if q.ndim != 4 or q.shape != k.shape or k.shape != v.shape:
        raise ValueError("expected matching BHND self-attention")
    if not 0 < density <= 1 or block < 1 or not 0 <= protected <= q.shape[-2]:
        raise ValueError("invalid density/block/protected")
    b,h,n,d = q.shape
    if n == 0:
        raise ValueError("empty sequence")
    scale = d ** -0.5
    nb = math.ceil(n/block)
    pad = nb*block-n
    kf = torch.nn.functional.pad(k.float(), (0,0,0,pad)).reshape(b,h,nb,block,d)
    vf = torch.nn.functional.pad(v.float(), (0,0,0,pad)).reshape_as(kf)
    valid = (torch.arange(nb*block, device=q.device)<n).reshape(nb,block)
    counts = valid.sum(-1).float()
    anchor = q.float().mean(-2)
    scores = (kf * anchor[:,:,None,None,:]).sum(-1)*scale
    scores = scores.masked_fill(~valid, float("-inf"))
    if weighted:
        weights = scores.softmax(-1)
    else:
        weights = valid.float()/counts[:,None]
    kb = (kf*weights[...,None]).sum(-2)
    vb = (vf*weights[...,None]).sum(-2)
    bias = (scores.logsumexp(-1)-(kb*anchor[:,:,None,:]).sum(-1)*scale
            if weighted else counts.log().expand(b,h,nb))
    out = torch.empty_like(q)
    top = min(nb,max(1,math.ceil(nb*density)))
    forced = math.ceil(protected/block)
    for start in range(0,n,block):
        z = q[:,:,start:start+block].float()
        summary = z @ kb.transpose(-1,-2)*scale+bias[:,:,None,:]
        route = summary.mean(-2)
        if forced:
            route[:,:,:forced] = float("inf")
        ids = route.topk(max(top,forced),dim=-1).indices
        ix = ids[...,None,None].expand(b,h,ids.shape[-1],block,d)
        ek = kf.gather(2,ix).flatten(2,3)
        ev = vf.gather(2,ix).flatten(2,3)
        mask = valid[ids].flatten(-2)
        exact = (z @ ek.transpose(-1,-2)*scale).masked_fill(~mask[:,:,None,:],float("-inf"))
        summary.scatter_(-1,ids[:,:,None,:].expand(b,h,z.shape[-2],ids.shape[-1]),float("-inf"))
        logits = torch.cat((exact,summary),dim=-1)
        p = logits.softmax(-1)
        m = ek.shape[-2]
        out[:,:,start:start+block] = (p[...,:m] @ ev + p[...,m:] @ vb).to(q.dtype)
    return out
