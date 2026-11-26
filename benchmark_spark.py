import argparse
import time
import torch
import torch.nn.functional as F
from spark_reference import sparse_reference

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--tokens",type=int,default=1021)
    p.add_argument("--heads",type=int,default=4)
    p.add_argument("--repeats",type=int,default=2)
    p.add_argument("--device",default="mps")
    a=p.parse_args()
    torch.manual_seed(42)
    def sync():
        if a.device=="mps": torch.mps.synchronize()
    def measure(fn):
        y=fn()
        sync()
        start=time.perf_counter()
        for _ in range(a.repeats):
            y=fn()
            sync()
        return y,(time.perf_counter()-start)/a.repeats
    with torch.inference_mode():
        q,k,v=[torch.randn(1,a.heads,a.tokens,128,device=a.device,dtype=torch.bfloat16) for _ in range(3)]
        ref,t=measure(lambda:F.scaled_dot_product_attention(q,k,v))
        print("dense seconds",t,flush=True)
        for density in (1.0,0.3,0.2,0.1):
            for weighted in (False,True):
                y,elapsed=measure(lambda:sparse_reference(q,k,v,density=density,weighted=weighted,protected=min(5,a.tokens)))
                yf,rf=y.float(),ref.float()
                rel=((yf-rf).norm()/rf.norm().clamp_min(1e-12)).item()
                cos=F.cosine_similarity(yf.flatten(),rf.flatten(),dim=0).item()
                finite=torch.isfinite(y).all().item()
                print("density",density,"weighted",weighted,"seconds",elapsed,"relative_l2",rel,"cosine",cos,"finite",finite,flush=True)
                if density==1 and (not finite or rel>0.02):
                    raise RuntimeError("full density reference mismatch")
if __name__=="__main__":
    main()
