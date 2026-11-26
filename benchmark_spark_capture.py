"""Offline benchmark of captured BHND attention tensors."""
import argparse
import json
import math
import time
from pathlib import Path
import torch
import torch.nn.functional as F
from safetensors import safe_open
from spark_reference import sparse_reference

@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("directory", type=Path)
    p.add_argument("--device", default="mps")
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--protected-blocks", type=int, default=5)
    p.add_argument("--block", type=int, default=64)
    p.add_argument("--output", type=Path, default=Path("spark_capture_report.json"))
    p.add_argument("--anchors", type=int, nargs="+", default=[0,1])
    a = p.parse_args()
    if any(i<0 or i>32 for i in a.anchors): p.error("anchors must be 0..32")
    if a.repeats < 1 or a.block < 1 or a.protected_blocks < 0:
        p.error("invalid repeats or block settings")
    files = sorted(a.directory.glob("*.safetensors"))
    if not files:
        p.error("no capture files in directory")
    def sync():
        if a.device == "mps":
            torch.mps.synchronize()
    def measure(fn):
        y = fn()
        sync()
        start = time.perf_counter()
        for _ in range(a.repeats):
            y = fn()
            sync()
        return y, (time.perf_counter()-start)/a.repeats
    report = {"torch":torch.__version__, "device":a.device,
              "block":a.block,"protected_blocks":a.protected_blocks,
              "repeats":a.repeats,"samples":[]}
    for path in files:
        with safe_open(str(path),framework="pt",device="cpu") as f:
            meta=f.metadata() or {}
            q,k,v=[f.get_tensor(name).to(a.device) for name in ("q","k","v")]
        if q.ndim!=4 or q.shape!=k.shape or k.shape!=v.shape:
            raise ValueError(f"invalid BHND capture: {path}")
        if meta.get("masked")!="false":
            raise ValueError(f"unsupported mask metadata: {path}")
        ref,seconds=measure(lambda:F.scaled_dot_product_attention(q,k,v))
        rf=ref.float()
        if not torch.isfinite(rf).all().item():
            raise RuntimeError("nonfinite dense baseline")
        n=q.shape[-2]
        protected=min(n,a.protected_blocks*a.block)
        sample={"file":path.name,"metadata":meta,"shape":list(q.shape),
                "dtype":str(q.dtype),"dense_seconds":seconds,"results":[]}
        print(f"\n{path.name} block={meta.get('block')} shape={tuple(q.shape)} dense={seconds:.6f}s",flush=True)
        for density in (1.0,0.5,0.3,0.2,0.1):
            for anchor_count in a.anchors:
                weighted = anchor_count > 0
                y,t=measure(lambda:sparse_reference(q,k,v,density=density,block=a.block,
                                                    weighted=weighted,protected=protected,anchors=max(1,anchor_count)))
                yf=y.float()
                finite=bool(torch.isfinite(yf).all().item())
                rel=((yf-rf).norm()/rf.norm().clamp_min(1e-12)).item()
                cos=F.cosine_similarity(yf.flatten(),rf.flatten(),dim=0).item()
                nb=math.ceil(n/a.block)
                row={"density":density,"effective_block_fraction":max(math.ceil(nb*density),math.ceil(protected/a.block))/nb,
                     "weighted":weighted,"anchors":anchor_count,"seconds":t,"relative_l2":rel if math.isfinite(rel) else None,
                     "cosine":cos if math.isfinite(cos) else None,"finite":finite}
                sample["results"].append(row)
                print(json.dumps(row),flush=True)
                if density==1 and (not finite or rel>0.02):
                    raise RuntimeError("full-density correctness check failed")
        report["samples"].append(sample)
        a.output.parent.mkdir(parents=True,exist_ok=True)
        a.output.write_text(json.dumps(report,indent=2,allow_nan=False))
        del q,k,v,ref,rf,y,yf
        if a.device=="mps":
            torch.mps.empty_cache()
    print("Report:",a.output.resolve())
if __name__=="__main__":
    main()
