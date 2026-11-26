"""Experimental MPS W8A8 SIMD-group GEMM. No CPU fallback or full weight expansion."""
import torch

_SHADER = r"""
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;
kernel void gemm(device const char* a [[buffer(0)]],
                 device const char* w [[buffer(1)]],
                 device const float* xs [[buffer(2)]],
                 device const float* ws [[buffer(3)]],
                 device bfloat* y [[buffer(4)]],
                 constant uint& k [[buffer(5)]],
                 constant uint& n [[buffer(6)]],
                 constant uint& m [[buffer(7)]],
                 uint2 tile [[threadgroup_position_in_grid]],
                 uint tid [[thread_index_in_threadgroup]],
                 uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup half aa[32*32], ww[32*32];
    threadgroup float tmp[32*32];
    int sums[8] = {};
    uint r0 = tile.y*32, c0 = tile.x*32;
    uint sr = (sg/2)*16, sc = (sg%2)*16;
    for (uint chunk=0; chunk<k; chunk+=256) {
        simdgroup_float8x8 acc[4];
        for (uint j=0; j<4; ++j)
            acc[j] = make_filled_simdgroup_matrix<float,8>(0.0f);
        for (uint kk=chunk; kk<min(chunk+256,k); kk+=32) {
            for (uint i=tid; i<1024; i+=128) {
                uint r=i/32, d=i%32;
                aa[i] = r0+r<m && kk+d<k ? half(a[ulong(r0+r)*k+kk+d]) : half(0);
                ww[i] = c0+r<n && kk+d<k ? half(w[ulong(c0+r)*k+kk+d]) : half(0);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint d=0; d<32; d+=8) {
                simdgroup_half8x8 av[2], wv[2];
                for (uint j=0; j<2; ++j) {
                    simdgroup_load(av[j], aa+(sr+j*8)*32+d, 32);
                    simdgroup_load(wv[j], ww+(sc+j*8)*32+d, 32, ulong2(0), true);
                }
                for (uint j=0; j<4; ++j)
                    simdgroup_multiply_accumulate(acc[j], av[j/2], wv[j%2], acc[j]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        for (uint j=0; j<4; ++j)
            simdgroup_store(acc[j], tmp+(sr+j/2*8)*32+sc+j%2*8, 32);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint j=0; j<8; ++j) sums[j] += int(tmp[tid+j*128]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint j=0; j<8; ++j) {
        uint i=tid+j*128, r=r0+i/32, c=c0+i%32;
        if (r<m && c<n) y[ulong(r)*n+c]=bfloat(float(sums[j])*xs[r]*ws[c]);
    }
}
"""

class MetalW8A8:
    def __init__(self):
        self.lib = None

    def compute(self, a, q, xs, ws):
        m,k = a.shape
        n = q.shape[0]
        if (a.device.type != "mps" or q.device != a.device
                or a.dtype != torch.int8 or q.dtype != torch.int8
                or q.shape[1] != k or not 0 < k <= 131072
                or not 0 < m*n < 2**31
                or xs.numel() != m or ws.numel() != n):
            raise ValueError("unsupported Metal W8A8 shape")
        xs = xs.float().reshape(-1).contiguous()
        ws = ws.float().reshape(-1).contiguous()
        for t in (a,q,xs,ws):
            if not t.is_contiguous() or t.storage_offset() != 0 or t.device != a.device:
                raise ValueError("unsupported Metal W8A8 buffer")
        if self.lib is None:
            self.lib = torch.mps.compile_shader(_SHADER)
        y = torch.empty((m,n), device=a.device, dtype=torch.bfloat16)
        self.lib.gemm(a,q,xs,ws,y,k,n,m,
                      threads=(((n+31)//32)*128,(m+31)//32),
                      group_size=(128,1))
        return y
