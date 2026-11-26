"""Capture real attention inputs without changing the active attention output."""
import logging
import threading
import uuid
from pathlib import Path
import torch
from safetensors.torch import save_file

def make_capture_override(previous, directory, head_count=2, max_samples=5, max_mib=256):
    directory = Path(directory) / uuid.uuid4().hex
    lock = threading.RLock()
    seen = set()
    calls = 0
    used = 0
    def override(func,q,k,v,heads,mask=None,attn_precision=None,skip_reshape=False,skip_output_reshape=False,**kwargs):
        nonlocal calls, used
        common = dict(mask=mask,attn_precision=attn_precision,skip_reshape=skip_reshape,
                      skip_output_reshape=skip_output_reshape,**kwargs)
        result = (func(q,k,v,heads,**common) if previous is None else
                  previous(func,q,k,v,heads,**common))
        with lock:
            calls += 1
            if len(seen)>=max_samples or mask is not None or q.shape!=k.shape or k.shape!=v.shape:
                return result
            options=kwargs.get("transformer_options") or {}
            block=options.get("sol_block",options.get("block_index"))
            try:
                block=int(block)
            except (TypeError,ValueError):
                block=None
            key=block if block is not None else calls
            if key in seen or (block is not None and block not in (2,12,25,37,49)):
                return result
            if block is None and calls not in (3,13,26,38,50):
                return result
            try:
                def view(x):
                    if skip_reshape:
                        if x.ndim!=4 or x.shape[1]!=heads: raise ValueError("unexpected BHND")
                        return x
                    if x.ndim!=3 or x.shape[-1]%heads: raise ValueError("unexpected BND")
                    return x.reshape(x.shape[0],x.shape[1],heads,-1).transpose(1,2)
                views=[view(x) for x in (q,k,v)]
                count=min(head_count,heads)
                size=sum(x[0:1,:count].numel()*x.element_size() for x in views)
                if used+size>max_mib*2**20: return result
                tensors={name:x[:1,:count].detach().to("cpu").contiguous().clone()
                         for name,x in zip(("q","k","v"),views)}
                directory.mkdir(parents=True,exist_ok=True)
                path=directory / f"attention_{calls:04d}.safetensors"
                save_file(tensors,str(path),metadata={"block":str(block),"call":str(calls),
                          "layout":"BHND","original_heads":str(heads),"masked":"false",
                          "head_indices":",".join(map(str,range(count)))})
                seen.add(key)
                used+=size
                logging.info("[H3 Capture] saved %s shape=%s block=%s",path,tuple(tensors["q"].shape),block)
            except Exception as exc:
                logging.warning("[H3 Capture] skipped: %r",exc)
        return result
    return override
