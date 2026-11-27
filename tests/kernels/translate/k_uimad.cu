// Kernels exercising sm_75 UIMAD with an immediate addend (uimad__URURsI_URURI, Issue #8) and
// 64-bit UIMAD.WIDE (uimad_wide__URURUR_URURUR) whose upper register half is read separately.
//   toolkits/cuda-12.9.2/bin/nvcc -allow-unsupported-compiler -arch=sm_70 -O3 -cubin -o k_uimad.sm70.cubin k_uimad.cu
//   toolkits/cuda-13.5.0/bin/nvcc -arch=sm_75 -O3 -cubin -o k_uimad.sm75.cubin k_uimad.cu
#include <stdint.h>

extern "C" __global__ void k_uimad_stride(
    uint32_t* out,
    const uint32_t* x,
    uint32_t a,
    uint32_t b,
    uint32_t stride,
    uint32_t n
) {
    uint32_t u0 = a * b - 7u;
    uint32_t u1 = (uint32_t)blockIdx.y * stride + 0x240u;
    const uint32_t* p = x + ((size_t)u0 * stride + (size_t)u1);
    uint32_t tid = threadIdx.x;
    uint32_t idx = blockIdx.x * blockDim.x + tid;
    if (idx < n) {
        out[idx] = p[tid] + u0 + u1;
    }
}

extern "C" __global__ void k_uimad_wide_hi(
    uint32_t* out32,
    uint64_t* out64,
    const uint32_t* in,
    uint32_t a,
    uint32_t b,
    uint64_t d,
    uint64_t e,
    int rounds,
    uint32_t n
) {
    uint32_t acc = (uint32_t)blockIdx.y * a + 0x240u;
    uint64_t w = (uint64_t)a * b + d;
    for (int i = 0; i < rounds; ++i) {
        w += (uint64_t)(a + (uint32_t)i) * b + e;
        acc += (uint32_t)(w >> 32) ^ (uint32_t)w;
    }
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        out32[tid] = acc + (uint32_t)(w >> 32) + in[tid];
        out64[tid] = w ^ (uint64_t)in[tid];
    }
}
