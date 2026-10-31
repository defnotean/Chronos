// ============================================================================
// Chronos-1T: Native CUDA 13.0 Fused Kernels for NVIDIA GB10 (sm_121)
// Target Platform: NVIDIA DGX Spark (GB10 Grace Blackwell Superchip)
//
// Ultra-Coalesced Vectorized Register-Tiled Synthesis:
//   - Uses 64-bit (`uint2` = 4 FP16 seeds -> 40 synthesized weights) and
//     128-bit (`uint4` = 8 FP16 activations) vector loads to saturate the
//     GB10's 228+ GB/s LPDDR5x bus with minimal L1 instruction pressure.
//   - Synthesizes 10 full-rank weights per compressed seed value directly in
//     SM registers via Chebyshev harmonic recurrence + signed ReLU^2.
// ============================================================================

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math_constants.h>
#include <stdint.h>

__device__ __forceinline__ float hss_register_activation(float z) {
    float pos = fmaxf(0.0f, z);
    float neg = fmaxf(0.0f, -z);
    return fmaf(pos, pos, fmaf(-0.25f, neg * neg, 0.12f * z));
}

template <int R_COMP = 10, int BLOCK_THREADS = 256, int ROWS_PER_WARP = 4>
__global__ void hss_10x_register_synthesis_kernel(
    const __half* __restrict__ compressed_seed, // [d_out, d_in / R_COMP] in LPDDR5x
    const float* __restrict__ spectral_core,    // [R_COMP, 4] IFFT-modulated Chebyshev kernel
    const __half* __restrict__ x_vec,           // [d_in] input token activation
    float* __restrict__ y_out,                  // [d_out] output activation
    int d_out,
    int d_in_compressed
) {
    // Load the 10x4 spectral-Chebyshev coefficients directly into registers/shared
    __shared__ float sh_c0[R_COMP];
    __shared__ float sh_c1[R_COMP];
    __shared__ float sh_c2[R_COMP];

    const int tid = threadIdx.x;
    if (tid < R_COMP) {
        sh_c0[tid] = spectral_core[tid * 4 + 0];
        sh_c1[tid] = spectral_core[tid * 4 + 1];
        sh_c2[tid] = spectral_core[tid * 4 + 2];
    }
    __syncthreads();

    float reg_c0[R_COMP];
    float reg_c1[R_COMP];
    float reg_c2[R_COMP];
    #pragma unroll
    for (int r = 0; r < R_COMP; ++r) {
        reg_c0[r] = sh_c0[r];
        reg_c1[r] = sh_c1[r];
        reg_c2[r] = sh_c2[r];
    }

    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    const int warps_per_block = BLOCK_THREADS / 32;
    const int base_row = (blockIdx.x * warps_per_block + warp_id) * ROWS_PER_WARP;

    if (base_row >= d_out) return;

    float acc[ROWS_PER_WARP];
    const uint2* row_ptrs_v4[ROWS_PER_WARP];
    #pragma unroll
    for (int m = 0; m < ROWS_PER_WARP; ++m) {
        acc[m] = 0.0f;
        int r_idx = (base_row + m < d_out) ? (base_row + m) : base_row;
        row_ptrs_v4[m] = reinterpret_cast<const uint2*>(
            compressed_seed + static_cast<size_t>(r_idx) * d_in_compressed
        );
    }

    // Each uint2 holds 4 FP16 seeds -> expands to 40 synthesized FP16 weights in registers!
    const int d_in_comp_v4 = d_in_compressed / 4;

    for (int col_v4 = lane_id; col_v4 < d_in_comp_v4; col_v4 += 32) {
        int base_x = col_v4 * (4 * R_COMP); // 40 FP16 elements = 80 bytes = 5 uint4 loads!
        const uint4* x_ptr_v8 = reinterpret_cast<const uint4*>(x_vec + base_x);

        // Load 40 FP16 input activations using just 5 coalesced 128-bit vector loads!
        uint4 xv0 = __ldg(&x_ptr_v8[0]);
        uint4 xv1 = __ldg(&x_ptr_v8[1]);
        uint4 xv2 = __ldg(&x_ptr_v8[2]);
        uint4 xv3 = __ldg(&x_ptr_v8[3]);
        uint4 xv4 = __ldg(&x_ptr_v8[4]);

        const __half2* xh2_0 = reinterpret_cast<const __half2*>(&xv0);
        const __half2* xh2_1 = reinterpret_cast<const __half2*>(&xv1);
        const __half2* xh2_2 = reinterpret_cast<const __half2*>(&xv2);
        const __half2* xh2_3 = reinterpret_cast<const __half2*>(&xv3);
        const __half2* xh2_4 = reinterpret_cast<const __half2*>(&xv4);

        float xr[4][R_COMP];
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            int idx0 = i * 2;
            xr[0][idx0]     = __low2float(xh2_0[i]);
            xr[0][idx0 + 1] = __high2float(xh2_0[i]);
        }
        xr[0][8] = __low2float(xh2_1[0]);
        xr[0][9] = __high2float(xh2_1[0]);

        xr[1][0] = __low2float(xh2_1[1]);
        xr[1][1] = __high2float(xh2_1[1]);
        xr[1][2] = __low2float(xh2_1[2]);
        xr[1][3] = __high2float(xh2_1[2]);
        xr[1][4] = __low2float(xh2_1[3]);
        xr[1][5] = __high2float(xh2_1[3]);
        xr[1][6] = __low2float(xh2_2[0]);
        xr[1][7] = __high2float(xh2_2[0]);
        xr[1][8] = __low2float(xh2_2[1]);
        xr[1][9] = __high2float(xh2_2[1]);

        xr[2][0] = __low2float(xh2_2[2]);
        xr[2][1] = __high2float(xh2_2[2]);
        xr[2][2] = __low2float(xh2_2[3]);
        xr[2][3] = __high2float(xh2_2[3]);
        xr[2][4] = __low2float(xh2_3[0]);
        xr[2][5] = __high2float(xh2_3[0]);
        xr[2][6] = __low2float(xh2_3[1]);
        xr[2][7] = __high2float(xh2_3[1]);
        xr[2][8] = __low2float(xh2_3[2]);
        xr[2][9] = __high2float(xh2_3[2]);

        xr[3][0] = __low2float(xh2_3[3]);
        xr[3][1] = __high2float(xh2_3[3]);
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            int idx0 = 2 + i * 2;
            xr[3][idx0]     = __low2float(xh2_4[i]);
            xr[3][idx0 + 1] = __high2float(xh2_4[i]);
        }

        // Now synthesize 40 weights per row across all 4 rows in registers!
        #pragma unroll
        for (int m = 0; m < ROWS_PER_WARP; ++m) {
            uint2 seed_v4 = __ldg(&row_ptrs_v4[m][col_v4]);
            const __half2* sh2 = reinterpret_cast<const __half2*>(&seed_v4);
            float s[4];
            s[0] = __low2float(sh2[0]);
            s[1] = __high2float(sh2[0]);
            s[2] = __low2float(sh2[1]);
            s[3] = __high2float(sh2[1]);

            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                float sk = s[k];
                float t1 = sk * 0.85f;
                float t2 = fmaf(2.0f * t1, t1, -1.0f);

                #pragma unroll
                for (int r = 0; r < R_COMP; ++r) {
                    float raw = fmaf(reg_c2[r], t2, fmaf(reg_c1[r], t1, reg_c0[r]));
                    float w = hss_register_activation(raw * sk);
                    acc[m] = fmaf(w, xr[k][r], acc[m]);
                }
            }
        }
    }

    // Warp butterfly reduction
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        #pragma unroll
        for (int m = 0; m < ROWS_PER_WARP; ++m) {
            acc[m] += __shfl_down_sync(0xffffffff, acc[m], offset);
        }
    }

    if (lane_id == 0) {
        #pragma unroll
        for (int m = 0; m < ROWS_PER_WARP; ++m) {
            if (base_row + m < d_out) {
                y_out[base_row + m] = acc[m];
            }
        }
    }
}

extern "C" {

void launch_hss_10x_register_synthesis(
    const void* compressed_seed,
    const float* spectral_core,
    const void* x_vec,
    float* y_out,
    int d_out,
    int d_in_compressed,
    cudaStream_t stream
) {
    constexpr int BLOCK_THREADS = 256;
    constexpr int WARPS_PER_BLOCK = BLOCK_THREADS / 32;
    constexpr int ROWS_PER_WARP = 4;
    constexpr int ROWS_PER_BLOCK = WARPS_PER_BLOCK * ROWS_PER_WARP;
    int grid = (d_out + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK;
    hss_10x_register_synthesis_kernel<10, BLOCK_THREADS, ROWS_PER_WARP><<<grid, BLOCK_THREADS, 0, stream>>>(
        reinterpret_cast<const __half*>(compressed_seed),
        spectral_core,
        reinterpret_cast<const __half*>(x_vec),
        y_out,
        d_out,
        d_in_compressed
    );
}

} // extern "C"
