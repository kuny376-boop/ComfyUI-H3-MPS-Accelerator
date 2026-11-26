"""Independent experimental KV reblocking; no runtime patches."""
import argparse
import json
import math
import time
from pathlib import Path
import torch
import torch.nn.functional as F
from safetensors import safe_open
from spark_reference import sparse_reference

def reorder(k,v,protected,mode):
    if mode=="fixed":
        return k,v
    # Balanced recursive median partition along highest-variance feature.
    # CPU indices for research only; preserve the protected prefix and KV pairing.
    x=k[:,:,protected:].float().cpu()
    if mode=="kv":
        y=v[:,:,protected:].float().cpu()
        x=torch.cat((x/x.square().mean(dim=(-2,-1),keepdim=True).sqrt().clamp_min(1e-8),
                     y/y.square().mean(dim=(-2,-1),keepdim=True).sqrt().clamp_min(1e-8)),dim=-1)
    def partition(z,ids):
        if ids.numel()<=64:
            return ids
        t=z[ids]
        axis=t.var(dim=0,unbiased=False).argmax().item()
        ids=ids[t[:,axis].argsort(stable=True)]
        mid=(ids.numel()//128)*64
        if mid==0: mid=64
        return torch.cat((partition(z,ids[:mid]),partition(z,ids[mid:])))
    order=torch.stack([torch.stack([partition(z,torch.arange(z.shape[0])) for z in batch]) for batch in x])
    prefix=torch.arange(protected).expand(*k.shape[:2],protected)
    order=torch.cat((prefix,order+protected),dim=-1).to(k.device)
    ix=order[...,None].expand_as(k)
    return k.gather(2,ix),v.gather(2,ix)

@torch.inference_mode()
def main():
    p=argparse.ArgumentParser()
    p.add_argument("directory",type=Path)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--device",default="mps")
    a=p.parse_args()
    def sync():
        if a.device=="mps": torch.mps.synchronize()
    def timed(fn):
        sync()
        t=time.perf_counter()
        y=fn()
        sync()
        return y,time.perf_counter()-t
    files=sorted(a.directory.glob("*.safetensors"))
    if not files: p.error("no captures")
    report=[]
    for path in files:
        with safe_open(str(path),framework="pt",device="cpu") as f:
            meta=f.metadata() or {}
            if meta.get("masked")!="false": raise ValueError("masked capture unsupported")
            q,k,v=[f.get_tensor(n).to(a.device) for n in ("q","k","v")]
        ref=F.scaled_dot_product_attention(q,k,v).float()
        rows=[]
        for mode,block in (("fixed",32),("fixed",64),("fixed",128),("k",64),("kv",64)):
            protected=min(q.shape[-2],math.ceil(320/block)*block)
            (kk,vv),prep=timed(lambda:reorder(k,v,protected,mode))
            for density in (1.,.5,.3,.2,.1):
                for weighted in (False,True):
                    fn=lambda:sparse_reference(q,kk,vv,density=density,block=block,weighted=weighted,protected=protected)
                    fn()
                    out,t=timed(fn)
                    out=out.float()
                    rel=((out-ref).norm()/ref.norm().clamp_min(1e-12)).item()
                    if not math.isfinite(rel) or (density==1 and rel>.02):
                        raise RuntimeError("correctness failure")
                    nb=math.ceil(q.shape[-2]/block)
                    row=dict(mode=mode,block=block,density=density,weighted=weighted,
                             effective_block_fraction=max(math.ceil(nb*density),math.ceil(protected/block))/nb,
                             protected_tokens=protected,relative_l2=rel,seconds=t,reblock_seconds=prep)
                    rows.append(row)
                    print(meta.get("block"),json.dumps(row),flush=True)
            del kk,vv
        report.append(dict(file=path.name,metadata=meta,results=rows))
        a.output.write_text(json.dumps(report,indent=2,allow_nan=False))
if __name__=="__main__":
    main()
