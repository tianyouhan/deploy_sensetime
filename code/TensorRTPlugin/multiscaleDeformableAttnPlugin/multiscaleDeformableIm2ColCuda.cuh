/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 **************************************************************************
 * Modified from Deformable DETR
 * Copyright (c) 2020-2023 SenseTime. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 [see LICENSE for details]
 * https://github.com/fundamentalvision/Deformable-DETR/blob/main/LICENSE
 **************************************************************************
 * Modified from DCN (https://github.com/msracver/Deformable-ConvNets)
 * Copyright (c) 2018-2023 Microsoft
 **************************************************************************
*/

#ifndef TRT_MULTISCALE_DEFORMABLE_IM2COL_CUDA_H
#define TRT_MULTISCALE_DEFORMABLE_IM2COL_CUDA_H

#include <algorithm>
#include <cstdio>
#include <cstring>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

// #include "common/checkMacrosPlugin.h"

#define CUDA_KERNEL_LOOP(i, n) for (int32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

constexpr int32_t kCUDA_NUM_THREADS{768};
inline int32_t GET_BLOCKS(int32_t const N, int32_t const numThreads)
{
    return (N + numThreads - 1) / numThreads;
}

template <typename scalar_t>
__device__ scalar_t ms_deform_attn_im2col_bilinear(scalar_t const*& bottomData, int32_t const& height, int32_t const& width,
    int32_t const& nHeads, int32_t const& channels, scalar_t const& h, scalar_t const& w, int32_t const& m, int32_t const& c)
{
    int32_t const hLow = floor(h);
    int32_t const wLow = floor(w);
    int32_t const hHigh = hLow + 1;
    int32_t const wHigh = wLow + 1;

    scalar_t const lh = h - hLow;
    scalar_t const lw = w - wLow;
    scalar_t const hh = 1 - lh, hw = 1 - lw;

    int32_t const wStride = nHeads * channels;
    int32_t const hStride = width * wStride;
    int32_t const hLowPtrOffset = hLow * hStride;
    int32_t const hHighPtrOffset = hLowPtrOffset + hStride;
    int32_t const wLowPtrOffset = wLow * wStride;
    int32_t const wHighPtrOffset = wLowPtrOffset + wStride;
    int32_t const basePtr = m * channels + c;

    scalar_t v1 = 0;
    if (hLow >= 0 && wLow >= 0)
    {
        int32_t const ptr1 = hLowPtrOffset + wLowPtrOffset + basePtr;
        v1 = bottomData[ptr1];
    }
    scalar_t v2 = 0;
    if (hLow >= 0 && wHigh <= width - 1)
    {
        int32_t const ptr2 = hLowPtrOffset + wHighPtrOffset + basePtr;
        v2 = bottomData[ptr2];
    }
    scalar_t v3 = 0;
    if (hHigh <= height - 1 && wLow >= 0)
    {
        int32_t const ptr3 = hHighPtrOffset + wLowPtrOffset + basePtr;
        v3 = bottomData[ptr3];
    }
    scalar_t v4 = 0;
    if (hHigh <= height - 1 && wHigh <= width - 1)
    {
        int32_t const ptr4 = hHighPtrOffset + wHighPtrOffset + basePtr;
        v4 = bottomData[ptr4];
    }

    scalar_t const w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;

    scalar_t const val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
    return val;
}

template <>
__device__ __half ms_deform_attn_im2col_bilinear<__half>(__half const*& bottomData, int32_t const& height, int32_t const& width,
    int32_t const& nHeads, int32_t const& channels, __half const& h, __half const& w, int32_t const& m, int32_t const& c)
{
    int32_t const hLow = __half2int_rd(h);
    int32_t const wLow = __half2int_rd(w);
    int32_t const hHigh = hLow + 1;
    int32_t const wHigh = wLow + 1;

    __half const kZERO = __int2half_rz(0);
    __half const one = __int2half_rz(1);

#if __CUDA_ARCH__>=530
    __half const lh = __hsub(h, __int2half_rd(hLow));
    __half const lw = __hsub(w, __int2half_rd(wLow));
    __half const hh = __hsub(one, lh), hw = __hsub(one, lw);
#else
    __half const lh = __float2half(__half2float(h) - hLow);
    __half const lw = __float2half(__half2float(w) - wLow);
    __half const hh = __float2half(__half2float(one) - __half2float(lh));
    __half const hw = __float2half(__half2float(one) - __half2float(lw));
#endif
    int32_t const wStride = nHeads * channels;
    int32_t const hStride = width * wStride;
    int32_t const hLowPtrOffset = hLow * hStride;
    int32_t const hHighPtrOffset = hLowPtrOffset + hStride;
    int32_t const wLowPtrOffset = wLow * wStride;
    int32_t const wHighPtrOffset = wLowPtrOffset + wStride;
    int32_t const basePtr = m * channels + c;

    __half v1 = kZERO;
    if (hLow >= 0 && wLow >= 0)
    {
        int32_t const ptr1 = hLowPtrOffset + wLowPtrOffset + basePtr;
        v1 = bottomData[ptr1];
    }
    __half v2 = kZERO;
    if (hLow >= 0 && wHigh <= width - 1)
    {
        int32_t const ptr2 = hLowPtrOffset + wHighPtrOffset + basePtr;
        v2 = bottomData[ptr2];
    }
    __half v3 = kZERO;
    if (hHigh <= height - 1 && wLow >= 0)
    {
        int32_t const ptr3 = hHighPtrOffset + wLowPtrOffset + basePtr;
        v3 = bottomData[ptr3];
    }
    __half v4 = kZERO;
    if (hHigh <= height - 1 && wHigh <= width - 1)
    {
        int32_t const ptr4 = hHighPtrOffset + wHighPtrOffset + basePtr;
        v4 = bottomData[ptr4];
    }

#if __CUDA_ARCH__>=530
    __half w1 = __hmul(__hmul(hh, hw), v1);
    __half w2 = __hmul(__hmul(hh, lw), v2);
    __half w3 = __hmul(__hmul(lh, hw), v3);
    __half w4 = __hmul(__hmul(lh, lw), v4);

    w1 = __hadd(w1, w2);
    w3 = __hadd(w3, w4);

    __half const val = __hadd(w1, w3);
#else
    __half w1 = __float2half((__half2float(hh) * __half2float(hw)) * __half2float(v1));
    __half w2 = __float2half((__half2float(hh) * __half2float(lw)) * __half2float(v2));
    __half w3 = __float2half((__half2float(lh) * __half2float(hw)) * __half2float(v3));
    __half w4 = __float2half((__half2float(lh) * __half2float(lw)) * __half2float(v4));
    
    w1 = __float2half(__half2float(w1) + __half2float(w2));
    w3 = __float2half(__half2float(w3) + __half2float(w4));

    __half const val = __float2half(__half2float(w1) + __half2float(w3));
#endif 
    return val;
}

// __device__ __half2 ms_deform_attn_im2col_bilinear_half2(
//     const __half* bottomData,         // pointer to FP16 data
//     int32_t const height,
//     int32_t const width,
//     int32_t const nHeads,
//     int32_t const channels,
//     __half const h,
//     __half const w,
//     int32_t const m,
//     int32_t const c0) // must be even: first channel of the pair
// {
//     // Determine integer cell coords
//     int32_t const hLow = __half2int_rd(h);
//     int32_t const wLow = __half2int_rd(w);
//     int32_t const hHigh = hLow + 1;
//     int32_t const wHigh = wLow + 1;

//     const __half kZERO = __int2half_rz(0);
//     const __half one = __int2half_rz(1);

//     // fractional parts
// #if __CUDA_ARCH__ >= 530
//     __half const lh = __hsub(h, __int2half_rd(hLow));
//     __half const lw = __hsub(w, __int2half_rd(wLow));
//     __half const hh = __hsub(one, lh);
//     __half const hw = __hsub(one, lw);
// #else
//     // fallback if half intrinsics not fully supported
//     __half const lh = __float2half(__half2float(h) - (float)hLow);
//     __half const lw = __float2half(__half2float(w) - (float)wLow);
//     __half const hh = __float2half(1.0f - __half2float(lh));
//     __half const hw = __float2half(1.0f - __half2float(lw));
// #endif

//     // strides/offsets (same as scalar)
//     int32_t const wStride = nHeads * channels;
//     int32_t const hStride = width * wStride;
//     int32_t const hLowPtrOffset = hLow * hStride;
//     int32_t const hHighPtrOffset = hLowPtrOffset + hStride;
//     int32_t const wLowPtrOffset = wLow * wStride;
//     int32_t const wHighPtrOffset = wLowPtrOffset + wStride;
//     int32_t const basePtr = m * channels + c0; // c0 is first channel of pair

//     // compute ptrs
//     int32_t ptr1 = hLowPtrOffset + wLowPtrOffset + basePtr;
//     int32_t ptr2 = hLowPtrOffset + wHighPtrOffset + basePtr;
//     int32_t ptr3 = hHighPtrOffset + wLowPtrOffset + basePtr;
//     int32_t ptr4 = hHighPtrOffset + wHighPtrOffset + basePtr;

//     // check that ptrs are even (they should be if channels is even)
//     // We'll index half2 array by ptr/2.
//     // Prepare a zero half2
//     #if __CUDA_ARCH__ >= 530
//     __half2 ZERO2 = __halves2half2(kZERO, kZERO); // duplicate zero
//     #else
//     // fallback: construct from two zeros (may compile on older toolchains too)
//     __half2 ZERO2 = *reinterpret_cast<const __half2*>(&kZERO); // risky fallback but usually ok
//     #endif

//     // reinterpret data pointer as half2 pointer (two consecutive half form one half2)
//     const __half2* bottom2 = reinterpret_cast<const __half2*>(bottomData);

//     // read v1..v4 as half2, set to ZERO2 if out-of-bounds
//     __half2 v1 = ZERO2, v2 = ZERO2, v3 = ZERO2, v4 = ZERO2;

//     if (hLow >= 0 && wLow >= 0) {
//         // ptr1 should be even; access half2 at ptr1/2
//         v1 = bottom2[ptr1 >> 1];
//     }
//     if (hLow >= 0 && wHigh <= width - 1) {
//         v2 = bottom2[ptr2 >> 1];
//     }
//     if (hHigh <= height - 1 && wLow >= 0) {
//         v3 = bottom2[ptr3 >> 1];
//     }
//     if (hHigh <= height - 1 && wHigh <= width - 1) {
//         v4 = bottom2[ptr4 >> 1];
//     }

//     // compute weights: hh*hw, hh*lw, lh*hw, lh*lw  (scalar half)
//     // then broadcast to half2 lanes (dup)
// #if __CUDA_ARCH__ >= 530
//     __half2 hh_hw = __halves2half2(__hmul(hh, hw), __hmul(hh, hw)); // weight for v1
//     __half2 hh_lw = __halves2half2(__hmul(hh, lw), __hmul(hh, lw)); // v2
//     __half2 lh_hw = __halves2half2(__hmul(lh, hw), __hmul(lh, hw)); // v3
//     __half2 lh_lw = __halves2half2(__hmul(lh, lw), __hmul(lh, lw)); // v4

//     // multiply and accumulate using half2 intrinsics: result = v1*hh_hw + v2*hh_lw + v3*lh_hw + v4*lh_lw
//     __half2 res = __hmul2(v1, hh_hw);
//     res = __hadd2(res, __hmul2(v2, hh_lw));
//     res = __hadd2(res, __hmul2(v3, lh_hw));
//     res = __hadd2(res, __hmul2(v4, lh_lw));
//     return res;
// #else
//     // Fallback path: do lane-wise scalar ops (less efficient) if half2 intrinsics unavailable
//     // extract lanes
//     __half v1_0 = __low2half(v1), v1_1 = __high2half(v1);
//     __half v2_0 = __low2half(v2), v2_1 = __high2half(v2);
//     __half v3_0 = __low2half(v3), v3_1 = __high2half(v3);
//     __half v4_0 = __low2half(v4), v4_1 = __high2half(v4);

//     __half w1 = __hmul(__hmul(hh, hw), kZERO); // temporary
//     // compute lane 0
//     __half sum0 = __hadd(__hadd(__hmul(__hmul(hh, hw), v1_0),
//                                 __hmul(__hmul(hh, lw), v2_0)),
//                          __hadd(__hmul(__hmul(lh, hw), v3_0),
//                                 __hmul(__hmul(lh, lw), v4_0)));
//     // lane1
//     __half sum1 = __hadd(__hadd(__hmul(__hmul(hh, hw), v1_1),
//                                 __hmul(__hmul(hh, lw), v2_1)),
//                          __hadd(__hmul(__hmul(lh, hw), v3_1),
//                                 __hmul(__hmul(lh, lw), v4_1)));
//     // pack back into half2
//     __half2 out;
//     out = __halves2half2(sum0, sum1);
//     return out;
// #endif
// }

__device__ __forceinline__
__half2 ms_deform_attn_im2col_bilinear_half2(
    const __half* __restrict__ bottomData,
    const int32_t height,
    const int32_t width,
    const int32_t nHeads,
    const int32_t channels,
    const __half  h,
    const __half  w,
    const int32_t m,
    const int32_t c
){
    const int32_t hLow  = __half2int_rd(h);
    const int32_t wLow  = __half2int_rd(w);
    const int32_t hHigh = hLow + 1;
    const int32_t wHigh = wLow + 1;

    const __half one = __int2half_rz(1);
    const __half lh = __hsub(h, __int2half_rd(hLow));
    const __half lw = __hsub(w, __int2half_rd(wLow));
    const __half hh = __hsub(one, lh);
    const __half hw = __hsub(one, lw);

    const __half2 hh2 = __halves2half2(hh, hh);
    const __half2 hw2 = __halves2half2(hw, hw);
    const __half2 lh2 = __halves2half2(lh, lh);
    const __half2 lw2 = __halves2half2(lw, lw);

    const int32_t wStride = nHeads * channels;
    const int32_t hStride = width * wStride;
    const int32_t hLowPtrOffset  = hLow  * hStride;
    const int32_t hHighPtrOffset = hLowPtrOffset + hStride;
    const int32_t wLowPtrOffset  = wLow  * wStride;
    const int32_t wHighPtrOffset = wLowPtrOffset + wStride;
    const int32_t basePtr = m * channels + c;

    const bool ok11 = (hLow  >= 0)           && (wLow  >= 0);
    const bool ok12 = (hLow  >= 0)           && (wHigh <= width  - 1);
    const bool ok21 = (hHigh <= height - 1)  && (wLow  >= 0);
    const bool ok22 = (hHigh <= height - 1)  && (wHigh <= width  - 1);

    __half2 v11 = __halves2half2(__int2half_rz(0), __int2half_rz(0));
    __half2 v12 = v11, v21 = v11, v22 = v11;

    if (ok11) {
        int32_t idx = hLowPtrOffset  + wLowPtrOffset  + basePtr;
        v11 = __halves2half2(bottomData[idx], bottomData[idx + 1]);
    }
    if (ok12) {
        int32_t idx = hLowPtrOffset  + wHighPtrOffset + basePtr;
        v12 = __halves2half2(bottomData[idx], bottomData[idx + 1]);
    }
    if (ok21) {
        int32_t idx = hHighPtrOffset + wLowPtrOffset  + basePtr;
        v21 = __halves2half2(bottomData[idx], bottomData[idx + 1]);
    }
    if (ok22) {
        int32_t idx = hHighPtrOffset + wHighPtrOffset + basePtr;
        v22 = __halves2half2(bottomData[idx], bottomData[idx + 1]);
    }

    // weights
    const __half2 w11 = __hmul2(hh2, hw2); // (1-lh)*(1-lw)
    const __half2 w12 = __hmul2(hh2, lw2);
    const __half2 w21 = __hmul2(lh2, hw2);
    const __half2 w22 = __hmul2(lh2, lw2);

    // compute four products first (same as scalar __half specialization does)
    __half2 p1 = __hmul2(w11, v11);
    __half2 p2 = __hmul2(w12, v12);
    __half2 p3 = __hmul2(w21, v21);
    __half2 p4 = __hmul2(w22, v22);

    // pairwise add to mimic scalar path: (p1+p2) + (p3+p4)
    __half2 s12 = __hadd2(p1, p2);
    __half2 s34 = __hadd2(p3, p4);
    __half2 acc = __hadd2(s12, s34);

    return acc;
}

__global__ void ms_deformable_im2col_gpu_kernel_half2(
    int64_t n_pairs,                        // half2 线程数 = floor(total_scalar / 2)
    const __half* __restrict__ dataValue,
    const int32_t* __restrict__ dataSpatialShapes,
    const int32_t* __restrict__ dataLevelStartIndex,
    const __half* __restrict__ dataSamplingLoc,
    const __half* __restrict__ dataAttnWeight,
    int32_t batchSize,
    int32_t spatialSize,
    int32_t numHeads,
    int32_t channels,
    int32_t numLevels,
    int32_t numQuery,
    int32_t numPoint,
    __half* __restrict__ dataCol)           // 输出（与 scalar layout 相同）
{
    const __half kZERO = __float2half(0.0f);
    const __half kHALF = __float2half(0.5f);
    const __half kMINUS_ONE = __float2half(-1.0f);

    int64_t total_scalar = (int64_t)batchSize * numQuery * numHeads * channels;
    int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * (int64_t)blockDim.x;

    for (; idx < n_pairs; idx += stride)
    {
        int64_t scalar_idx0 = idx * 2;
        int64_t scalar_idx1 = scalar_idx0 + 1;
        if (scalar_idx0 >= total_scalar) continue;

        // decode scalar_idx0 -> (bCol0, qid0, mCol0, c0)
        int64_t tmp0 = scalar_idx0;
        int32_t c0   = (int32_t)(tmp0 % channels); tmp0 /= channels;
        int32_t mCol0 = (int32_t)(tmp0 % numHeads); tmp0 /= numHeads;
        int32_t qid0  = (int32_t)(tmp0 % numQuery); tmp0 /= numQuery;
        int32_t bCol0 = (int32_t)tmp0;

        // decode scalar_idx1 if valid and ensure same group
        bool has1 = (scalar_idx1 < total_scalar);
        int32_t c1=0, mCol1=0, qid1=0, bCol1=0;
        if (has1) {
            int64_t tmp1 = scalar_idx1;
            c1 = (int32_t)(tmp1 % channels); tmp1 /= channels;
            mCol1 = (int32_t)(tmp1 % numHeads); tmp1 /= numHeads;
            qid1 = (int32_t)(tmp1 % numQuery); tmp1 /= numQuery;
            bCol1 = (int32_t)tmp1;
            if (!(bCol1==bCol0 && qid1==qid0 && mCol1==mCol0)) {
                has1 = false;
            }
        }

        __half acc0 = kZERO;
        __half acc1 = kZERO;

        // include batch & qid in samplingIndex
        int32_t samplingIndex = ((int32_t)bCol0 * numQuery + qid0) * numHeads + mCol0;

        int32_t qidStride = numHeads * channels;
        int32_t dataValuePtrInitOffset = bCol0 * spatialSize * qidStride;

        int32_t dataWeightPtr = samplingIndex * numLevels * numPoint;
        int32_t dataLocWPtr  = dataWeightPtr << 1;

        for (int32_t l=0; l < numLevels; ++l) {
            int32_t levelStartId = dataLevelStartIndex[l];
            int32_t spatialH = dataSpatialShapes[l*2 + 0];
            int32_t spatialW = dataSpatialShapes[l*2 + 1];
            __half spatialHHalf = __int2half_rd(spatialH);
            __half spatialWHalf = __int2half_rd(spatialW);
            const __half* dataValuePtr = dataValue + (dataValuePtrInitOffset + levelStartId * qidStride);

            for (int32_t p=0; p < numPoint; ++p) {
                __half locW = dataSamplingLoc[dataLocWPtr];
                __half locH = dataSamplingLoc[dataLocWPtr + 1];
                __half weight = dataAttnWeight[dataWeightPtr];

                __half hIm = __hsub(__hmul(locH, spatialHHalf), kHALF);
                __half wIm = __hsub(__hmul(locW, spatialWHalf), kHALF);

                bool inside = (__hgt(hIm, kMINUS_ONE) && __hgt(wIm, kMINUS_ONE)
                               && __hlt(hIm, spatialHHalf) && __hlt(wIm, spatialWHalf));
                if (inside) {
                    __half2 pair = ms_deform_attn_im2col_bilinear_half2(
                        dataValuePtr, spatialH, spatialW, numHeads, channels, hIm, wIm, mCol0, c0);

                    #if __CUDA_ARCH__ >= 530
                    __half2 w2 = __halves2half2(weight, weight);
                    __half2 mul = __hmul2(pair, w2);
                    #else
                    __half t0 = __hmul(__low2half(pair), weight);
                    __half t1 = __hmul(__high2half(pair), weight);
                    __half2 mul = __halves2half2(t0, t1);
                    #endif

                    acc0 = __hadd(acc0, __low2half(mul));
                    acc1 = __hadd(acc1, __high2half(mul));
                }

                dataWeightPtr += 1;
                dataLocWPtr  += 2;
            }
        }

        // 写回结果（与 scalar layout 一致）
        int64_t out_index0 = ((int64_t)bCol0 * numQuery * numHeads * channels) +
                             ((int64_t)qid0 * numHeads * channels) +
                             ((int64_t)mCol0 * channels) + c0;
        dataCol[out_index0] = acc0;
        if (has1 && c1 < channels) dataCol[out_index0 + 1] = acc1;
    }
}

// ===================== host 启动器 =====================
inline void ms_deformable_im2col_cuda_half2(
    cudaStream_t stream,
    const __half* dataValue,
    const int32_t* dataSpatialShapes,
    const int32_t* dataLevelStartIndex,
    const __half* dataSamplingLoc,
    const __half* dataAttnWeight,
    int32_t batchSize, int32_t spatialSize,
    int32_t numHeads, int32_t channels,
    int32_t numLevels, int32_t numQuery, int32_t numPoint,
    __half* dataCol)
{
    // printf("start ms_deformable_im2col_cuda_half2");
    if(channels % 2 != 0)
    {
        printf("ERROR: channels(%d) must be even!\n", channels);
        return;
    }

    int64_t numKernels = (int64_t)batchSize * (int64_t)numQuery * (int64_t)numHeads * (channels/2);
    int threads = 256;
    int blocks = (numKernels + threads - 1) / threads;

    ms_deformable_im2col_gpu_kernel_half2<<<blocks, threads, 0, stream>>>(
        numKernels, dataValue, dataSpatialShapes, dataLevelStartIndex,
        dataSamplingLoc, dataAttnWeight,
        batchSize, spatialSize, numHeads, channels,
        numLevels, numQuery, numPoint, dataCol);

    cudaError_t err = cudaGetLastError();
    if(err != cudaSuccess) {
        printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
    }
    // printf("end ms_deformable_im2col_cuda_half2");
}


template <typename scalar_t>
__device__ void ms_deform_attn_col2im_bilinear(scalar_t const*& bottomData, int32_t const& height, int32_t const& width,
    int32_t const& nHeads, int32_t const& channels, scalar_t const& h, scalar_t const& w, int32_t const& m, int32_t const& c,
    scalar_t const& topGrad, scalar_t const& attnWeight, scalar_t*& gradValue, scalar_t* gradSamplingLoc,
    scalar_t* gradAttnWeight)
{
    int32_t const hLow = floor(h);
    int32_t const wLow = floor(w);
    int32_t const hHigh = hLow + 1;
    int32_t const wHigh = wLow + 1;

    scalar_t const lh = h - hLow;
    scalar_t const lw = w - wLow;
    scalar_t const hh = 1 - lh, hw = 1 - lw;

    int32_t const wStride = nHeads * channels;
    int32_t const hStride = width * wStride;
    int32_t const hLowPtrOffset = hLow * hStride;
    int32_t const hHighPtrOffset = hLowPtrOffset + hStride;
    int32_t const wLowPtrOffset = wLow * wStride;
    int32_t const wHighPtrOffset = wLowPtrOffset + wStride;
    int32_t const basePtr = m * channels + c;

    scalar_t const w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;
    scalar_t const topGradvalue = topGrad * attnWeight;
    scalar_t gradHWeight = 0, gradWWeight = 0;

    scalar_t v1 = 0;
    if (hLow >= 0 && wLow >= 0)
    {
        int32_t const ptr1 = hLowPtrOffset + wLowPtrOffset + basePtr;
        v1 = bottomData[ptr1];
        gradHWeight -= hw * v1;
        gradWWeight -= hh * v1;
        atomicAdd(gradValue + ptr1, w1 * topGradvalue);
    }
    scalar_t v2 = 0;
    if (hLow >= 0 && wHigh <= width - 1)
    {
        int32_t const ptr2 = hLowPtrOffset + wHighPtrOffset + basePtr;
        v2 = bottomData[ptr2];
        gradHWeight -= lw * v2;
        gradWWeight += hh * v2;
        atomicAdd(gradValue + ptr2, w2 * topGradvalue);
    }
    scalar_t v3 = 0;
    if (hHigh <= height - 1 && wLow >= 0)
    {
        int32_t const ptr3 = hHighPtrOffset + wLowPtrOffset + basePtr;
        v3 = bottomData[ptr3];
        gradHWeight += hw * v3;
        gradWWeight -= lh * v3;
        atomicAdd(gradValue + ptr3, w3 * topGradvalue);
    }
    scalar_t v4 = 0;
    if (hHigh <= height - 1 && wHigh <= width - 1)
    {
        int32_t const ptr4 = hHighPtrOffset + wHighPtrOffset + basePtr;
        v4 = bottomData[ptr4];
        gradHWeight += lw * v4;
        gradWWeight += lh * v4;
        atomicAdd(gradValue + ptr4, w4 * topGradvalue);
    }

    scalar_t const val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
    *gradAttnWeight = topGrad * val;
    *gradSamplingLoc = width * gradWWeight * topGradvalue;
    *(gradSamplingLoc + 1) = height * gradHWeight * topGradvalue;
}

template <typename scalar_t>
__device__ void ms_deform_attn_col2im_bilinear_gm(scalar_t const*& bottomData, int32_t const& height, int32_t const& width,
    int32_t const& nHeads, int32_t const& channels, scalar_t const& h, scalar_t const& w, int32_t const& m, int32_t const& c,
    scalar_t const& topGrad, scalar_t const& attnWeight, scalar_t*& gradValue, scalar_t* gradSamplingLoc,
    scalar_t* gradAttnWeight)
{
    int32_t const hLow = floor(h);
    int32_t const wLow = floor(w);
    int32_t const hHigh = hLow + 1;
    int32_t const wHigh = wLow + 1;

    scalar_t const lh = h - hLow;
    scalar_t const lw = w - wLow;
    scalar_t const hh = 1 - lh, hw = 1 - lw;

    int32_t const wStride = nHeads * channels;
    int32_t const hStride = width * wStride;
    int32_t const hLowPtrOffset = hLow * hStride;
    int32_t const hHighPtrOffset = hLowPtrOffset + hStride;
    int32_t const wLowPtrOffset = wLow * wStride;
    int32_t const wHighPtrOffset = wLowPtrOffset + wStride;
    int32_t const basePtr = m * channels + c;

    scalar_t const w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;
    scalar_t const topGradvalue = topGrad * attnWeight;
    scalar_t gradHWeight = 0, gradWWeight = 0;

    scalar_t v1 = 0;
    if (hLow >= 0 && wLow >= 0)
    {
        int32_t const ptr1 = hLowPtrOffset + wLowPtrOffset + basePtr;
        v1 = bottomData[ptr1];
        gradHWeight -= hw * v1;
        gradWWeight -= hh * v1;
        atomicAdd(gradValue + ptr1, w1 * topGradvalue);
    }
    scalar_t v2 = 0;
    if (hLow >= 0 && wHigh <= width - 1)
    {
        int32_t const ptr2 = hLowPtrOffset + wHighPtrOffset + basePtr;
        v2 = bottomData[ptr2];
        gradHWeight -= lw * v2;
        gradWWeight += hh * v2;
        atomicAdd(gradValue + ptr2, w2 * topGradvalue);
    }
    scalar_t v3 = 0;
    if (hHigh <= height - 1 && wLow >= 0)
    {
        int32_t const ptr3 = hHighPtrOffset + wLowPtrOffset + basePtr;
        v3 = bottomData[ptr3];
        gradHWeight += hw * v3;
        gradWWeight -= lh * v3;
        atomicAdd(gradValue + ptr3, w3 * topGradvalue);
    }
    scalar_t v4 = 0;
    if (hHigh <= height - 1 && wHigh <= width - 1)
    {
        int32_t const ptr4 = hHighPtrOffset + wHighPtrOffset + basePtr;
        v4 = bottomData[ptr4];
        gradHWeight += lw * v4;
        gradWWeight += lh * v4;
        atomicAdd(gradValue + ptr4, w4 * topGradvalue);
    }

    scalar_t const val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
    atomicAdd(gradAttnWeight, topGrad * val);
    atomicAdd(gradSamplingLoc, width * gradWWeight * topGradvalue);
    atomicAdd(gradSamplingLoc + 1, height * gradHWeight * topGradvalue);
}

#if 1
template <typename scalar_t>
__global__ void ms_deformable_im2col_gpu_kernel(int32_t const n, scalar_t const* dataValue,
    int32_t const* dataSpatialShapes, int32_t const* dataLevelStartIndex, scalar_t const* dataSamplingLoc,
    scalar_t const* dataAttnWeight, int32_t const batchSize, int32_t const spatialSize, int32_t const numHeads, int32_t const channels,
    int32_t const numLevels, int32_t const numQuery, int32_t const numPoint, scalar_t* dataCol)
{
    CUDA_KERNEL_LOOP(index, n)
    {
        int32_t _temp = index;
        int32_t const cCol = _temp % channels;
        _temp /= channels;
        int32_t const samplingIndex = _temp;
        int32_t const mCol = _temp % numHeads;
        _temp /= numHeads;
        _temp /= numQuery;
        int32_t const bCol = _temp;

        scalar_t* dataColPtr = dataCol + index;
        int32_t dataWeightPtr = samplingIndex * numLevels * numPoint;
        int32_t dataLocWPtr = dataWeightPtr << 1;
        int32_t const qidStride = numHeads * channels;
        int32_t const dataValuePtrInitOffset = bCol * spatialSize * qidStride;
        scalar_t col = 0;

        for (int32_t lCol = 0; lCol < numLevels; ++lCol)
        {
            int32_t const levelStartId = dataLevelStartIndex[lCol];
            int32_t const spatialHPtr = lCol << 1;
            int32_t const spatialH = dataSpatialShapes[spatialHPtr];
            int32_t const spatialW = dataSpatialShapes[spatialHPtr + 1];
            scalar_t const* dataValuePtr = dataValue + (dataValuePtrInitOffset + levelStartId * qidStride);
            for (int32_t pCol = 0; pCol < numPoint; ++pCol)
            {
                scalar_t const locW = dataSamplingLoc[dataLocWPtr];
                scalar_t const locH = dataSamplingLoc[dataLocWPtr + 1];
                scalar_t const weight = dataAttnWeight[dataWeightPtr];

                scalar_t const hIm = locH * spatialH - 0.5;
                scalar_t const wIm = locW * spatialW - 0.5;

                if (hIm > -1 && wIm > -1 && hIm < spatialH && wIm < spatialW)
                {
                    col += ms_deform_attn_im2col_bilinear(
                               dataValuePtr, spatialH, spatialW, numHeads, channels, hIm, wIm, mCol, cCol)
                        * weight;
                }

                dataWeightPtr += 1;
                dataLocWPtr += 2;
            }
        }
        *dataColPtr = col;
    }
}

template <>
__global__ void ms_deformable_im2col_gpu_kernel<__half>(int32_t const n, __half const* dataValue,
    int32_t const* dataSpatialShapes, int32_t const* dataLevelStartIndex, __half const* dataSamplingLoc,
    __half const* dataAttnWeight, int32_t const batchSize, int32_t const spatialSize, int32_t const numHeads, int32_t const channels,
    int32_t const numLevels, int32_t const numQuery, int32_t const numPoint, __half* dataCol)
{
    CUDA_KERNEL_LOOP(index, n)
    {
        int32_t _temp = index;
        int32_t const cCol = _temp % channels;
        _temp /= channels;
        int32_t const samplingIndex = _temp;
        int32_t const mCol = _temp % numHeads;
        _temp /= numHeads;
        _temp /= numQuery;
        int32_t const bCol = _temp;

        __half* dataColPtr = dataCol + index;
        int32_t dataWeightPtr = samplingIndex * numLevels * numPoint;
        int32_t dataLocWPtr = dataWeightPtr << 1;
        int32_t const qidStride = numHeads * channels;
        int32_t const dataValuePtrInitOffset = bCol * spatialSize * qidStride;
        __half const kZERO_POINT_FIVE = __float2half(0.5f);
        __half const kMINUS_ONE = __float2half(-1.0f);
        __half const kZERO = __int2half_rz(0);
        __half tpVal = kZERO;
        __half col = kZERO;

        for (int32_t lCol = 0; lCol < numLevels; ++lCol)
        {
            int32_t const levelStartId = dataLevelStartIndex[lCol];
            int32_t const spatialHPtr = lCol << 1;
            int32_t const spatialH = dataSpatialShapes[spatialHPtr];
            int32_t const spatialW = dataSpatialShapes[spatialHPtr + 1];
            __half const spatialHHalf = __int2half_rd(spatialH);
            __half const spatialWHalf = __int2half_rd(spatialW);
            __half const* dataValuePtr = dataValue + (dataValuePtrInitOffset + levelStartId * qidStride);
            for (int32_t pCol = 0; pCol < numPoint; ++pCol)
            {
                __half const locW = dataSamplingLoc[dataLocWPtr];
                __half const locH = dataSamplingLoc[dataLocWPtr + 1];
                __half const weight = dataAttnWeight[dataWeightPtr];
#if __CUDA_ARCH__ >= 530
                __half const hIm = __hsub(__hmul(locH, spatialHHalf), kZERO_POINT_FIVE);
                __half const wIm = __hsub(__hmul(locW, spatialWHalf), kZERO_POINT_FIVE);

                if (__hgt(hIm, kMINUS_ONE) && __hgt(wIm, kMINUS_ONE) && __hlt(hIm, spatialHHalf)
                    && __hlt(wIm, spatialWHalf))
                {
                    tpVal = ms_deform_attn_im2col_bilinear(
                        dataValuePtr, spatialH, spatialW, numHeads, channels, hIm, wIm, mCol, cCol);
                    col = __hadd(col, __hmul(tpVal, weight));
                }
#else
                __half const hIm = __float2half(__half2float(locH) * __half2float(spatialHHalf) - __half2float(kZERO_POINT_FIVE));
                __half const wIm = __float2half(__half2float(locW) * __half2float(spatialWHalf) - __half2float(kZERO_POINT_FIVE));

                if((__half2float(hIm)>__half2float(kMINUS_ONE)) && (__half2float(wIm)>__half2float(kMINUS_ONE))
                    && (__half2float(hIm)<__half2float(spatialHHalf)) && (__half2float(wIm)<__half2float(spatialWHalf)))
                {
                    tpVal = ms_deform_attn_im2col_bilinear(
                        dataValuePtr, spatialH, spatialW, numHeads, channels, hIm, wIm, mCol, cCol);
                    col = __float2half(__half2float(col) + (__half2float(tpVal) * __half2float(weight)));
                }
#endif 
                dataWeightPtr += 1;
                dataLocWPtr += 2;
            }
        }
        *dataColPtr = col;
    }
}
#endif

template <typename scalar_t, uint32_t blockSize>
__global__ void ms_deformable_col2im_gpu_kernel_shm_blocksize_aware_reduce_v1(int32_t const n, scalar_t const* grad_col,
    scalar_t const* dataValue, int32_t const* dataSpatialShapes, int32_t const* dataLevelStartIndex,
    scalar_t const* dataSamplingLoc, scalar_t const* dataAttnWeight, int32_t const batchSize, int32_t const spatialSize,
    int32_t const numHeads, int32_t const channels, int32_t const numLevels, int32_t const numQuery, int32_t const numPoint,
    scalar_t* gradValue, scalar_t* gradSamplingLoc, scalar_t* gradAttnWeight)
{
    CUDA_KERNEL_LOOP(index, n)
    {
        __shared__ scalar_t cacheGradSamplingLoc[blockSize * 2];
        __shared__ scalar_t cacheGradAttnWeight[blockSize];
        uint32_t tid = threadIdx.x;
        int32_t _temp = index;
        int32_t const cCol = _temp % channels;
        _temp /= channels;
        int32_t const samplingIndex = _temp;
        int32_t const mCol = _temp % numHeads;
        _temp /= numHeads;
        int32_t const qCol = _temp % numQuery;
        _temp /= numQuery;
        int32_t const bCol = _temp;

        scalar_t const topGrad = grad_col[index];

        int32_t dataWeightPtr = samplingIndex * numLevels * numPoint;
        int32_t dataLocWPtr = dataWeightPtr << 1;
        int32_t const gradSamplingPtr = dataWeightPtr;
        gradSamplingLoc += gradSamplingPtr << 1;
        gradAttnWeight += gradSamplingPtr;
        int32_t const gradWeightStride = 1;
        int32_t const gradLocStride = 2;
        int32_t const qidStride = numHeads * channels;
        int32_t const dataValuePtrInitOffset = bCol * spatialSize * qidStride;

        for (int32_t lCol = 0; lCol < numLevels; ++lCol)
        {
            int32_t const levelStartId = dataLevelStartIndex[lCol];
            int32_t const spatialHPtr = lCol << 1;
            int32_t const spatialH = dataSpatialShapes[spatialHPtr];
            int32_t const spatialW = dataSpatialShapes[spatialHPtr + 1];
            int32_t const valuePtrOffset = dataValuePtrInitOffset + levelStartId * qidStride;
            scalar_t const* dataValuePtr = dataValue + valuePtrOffset;
            scalar_t* gradValuePtr = gradValue + valuePtrOffset;

            for (int32_t pCol = 0; pCol < numPoint; ++pCol)
            {
                scalar_t const locW = dataSamplingLoc[dataLocWPtr];
                scalar_t const locH = dataSamplingLoc[dataLocWPtr + 1];
                scalar_t const weight = dataAttnWeight[dataWeightPtr];

                scalar_t const hIm = locH * spatialH - 0.5;
                scalar_t const wIm = locW * spatialW - 0.5;
                *(cacheGradSamplingLoc + (threadIdx.x << 1)) = 0;
                *(cacheGradSamplingLoc + ((threadIdx.x << 1) + 1)) = 0;
                *(cacheGradAttnWeight + threadIdx.x) = 0;
                if (hIm > -1 && wIm > -1 && hIm < spatialH && wIm < spatialW)
                {
                    ms_deform_attn_col2im_bilinear(dataValuePtr, spatialH, spatialW, numHeads, channels, hIm, wIm, mCol,
                        cCol, topGrad, weight, gradValuePtr, cacheGradSamplingLoc + (threadIdx.x << 1),
                        cacheGradAttnWeight + threadIdx.x);
                }

                __syncthreads();
                if (tid == 0)
                {
                    scalar_t _gradW = cacheGradSamplingLoc[0], _gradH = cacheGradSamplingLoc[1],
                             _gradA = cacheGradAttnWeight[0];
                    int32_t sid = 2;
                    for (uint32_t tid = 1; tid < blockSize; ++tid)
                    {
                        _gradW += cacheGradSamplingLoc[sid];
                        _gradH += cacheGradSamplingLoc[sid + 1];
                        _gradA += cacheGradAttnWeight[tid];
                        sid += 2;
                    }

                    *gradSamplingLoc = _gradW;
                    *(gradSamplingLoc + 1) = _gradH;
                    *gradAttnWeight = _gradA;
                }
                __syncthreads();

                dataWeightPtr += 1;
                dataLocWPtr += 2;
                gradAttnWeight += gradWeightStride;
                gradSamplingLoc += gradLocStride;
            }
        }
    }
}

template <typename scalar_t, uint32_t blockSize>
__global__ void ms_deformable_col2im_gpu_kernel_shm_blocksize_aware_reduce_v2(int32_t const n, scalar_t const* grad_col,
    scalar_t const* dataValue, int32_t const* dataSpatialShapes, int32_t const* dataLevelStartIndex,
    scalar_t const* dataSamplingLoc, scalar_t const* dataAttnWeight, int32_t const batchSize, int32_t const spatialSize,
    int32_t const numHeads, int32_t const channels, int32_t const numLevels, int32_t const numQuery, int32_t const numPoint,
    scalar_t* gradValue, scalar_t* gradSamplingLoc, scalar_t* gradAttnWeight)
{
    CUDA_KERNEL_LOOP(index, n)
    {
        __shared__ scalar_t cacheGradSamplingLoc[blockSize * 2];
        __shared__ scalar_t cacheGradAttnWeight[blockSize];
        uint32_t tid = threadIdx.x;
        int32_t _temp = index;
        int32_t const cCol = _temp % channels;
        _temp /= channels;
        int32_t const samplingIndex = _temp;
        int32_t const mCol = _temp % numHeads;
        _temp /= numHeads;
        int32_t const qCol = _temp % numQuery;
        _temp /= numQuery;
        int32_t const bCol = _temp;

        scalar_t const topGrad = grad_col[index];

        int32_t dataWeightPtr = samplingIndex * numLevels * numPoint;
        int32_t dataLocWPtr = dataWeightPtr << 1;
        int32_t const gradSamplingPtr = dataWeightPtr;
        gradSamplingLoc += gradSamplingPtr << 1;
        gradAttnWeight += gradSamplingPtr;
        int32_t const gradWeightStride = 1;
        int32_t const gradLocStride = 2;
        int32_t const qidStride = numHeads * channels;
        int32_t const dataValuePtrInitOffset = bCol * spatialSize * qidStride;

        for (int32_t lCol = 0; lCol < numLevels; ++lCol)
        {
            int32_t const levelStartId = dataLevelStartIndex[lCol];
            int32_t const spatialHPtr = lCol << 1;
            int32_t const spatialH = dataSpatialShapes[spatialHPtr];
            int32_t const spatialW = dataSpatialShapes[spatialHPtr + 1];
            int32_t const valuePtrOffset = dataValuePtrInitOffset + levelStartId * qidStride;
            scalar_t const* dataValuePtr = dataValue + valuePtrOffset;
            scalar_t* gradValuePtr = gradValue + valuePtrOffset;

            for (int32_t pCol = 0; pCol < numPoint; ++pCol)
            {
                scalar_t const locW = dataSamplingLoc[dataLocWPtr];
                scalar_t const locH = dataSamplingLoc[dataLocWPtr + 1];
                scalar_t const weight = dataAttnWeight[dataWeightPtr];

                scalar_t const hIm = locH * spatialH - 0.5;
                scalar_t const wIm = locW * spatialW - 0.5;
                *(cacheGradSamplingLoc + (threadIdx.x << 1)) = 0;
                *(cacheGradSamplingLoc + ((threadIdx.x << 1) + 1)) = 0;
                *(cacheGradAttnWeight + threadIdx.x) = 0;
                if (hIm > -1 && wIm > -1 && hIm < spatialH && wIm < spatialW)
                {
                    ms_deform_attn_col2im_bilinear(dataValuePtr, spatialH, spatialW, numHeads, channels, hIm, wIm, mCol,
                        cCol, topGrad, weight, gradValuePtr, cacheGradSamplingLoc + (threadIdx.x << 1),
                        cacheGradAttnWeight + threadIdx.x);
                }

                __syncthreads();

                for (uint32_t s = blockSize / 2; s > 0; s >>= 1)
                {
                    if (tid < s)
                    {
                        uint32_t const xid1 = tid << 1;
                        uint32_t const xid2 = (tid + s) << 1;
                        cacheGradAttnWeight[tid] += cacheGradAttnWeight[tid + s];
                        cacheGradSamplingLoc[xid1] += cacheGradSamplingLoc[xid2];
                        cacheGradSamplingLoc[xid1 + 1] += cacheGradSamplingLoc[xid2 + 1];
                    }
                    __syncthreads();
                }

                if (tid == 0)
                {
                    *gradSamplingLoc = cacheGradSamplingLoc[0];
                    *(gradSamplingLoc + 1) = cacheGradSamplingLoc[1];
                    *gradAttnWeight = cacheGradAttnWeight[0];
                }
                __syncthreads();

                dataWeightPtr += 1;
                dataLocWPtr += 2;
                gradAttnWeight += gradWeightStride;
                gradSamplingLoc += gradLocStride;
            }
        }
    }
}

template <typename scalar_t>
__global__ void ms_deformable_col2im_gpu_kernel_shm_reduce_v1(int32_t const n, scalar_t const* grad_col,
    scalar_t const* dataValue, int32_t const* dataSpatialShapes, int32_t const* dataLevelStartIndex,
    scalar_t const* dataSamplingLoc, scalar_t const* dataAttnWeight, int32_t const batchSize, int32_t const spatialSize,
    int32_t const numHeads, int32_t const channels, int32_t const numLevels, int32_t const numQuery, int32_t const numPoint,
    scalar_t* gradValue, scalar_t* gradSamplingLoc, scalar_t* gradAttnWeight)
{
    CUDA_KERNEL_LOOP(index, n)
    {
        extern __shared__ int32_t _s[];
        scalar_t* cacheGradSamplingLoc = (scalar_t*) _s;
        scalar_t* cacheGradAttnWeight = cacheGradSamplingLoc + 2 * blockDim.x;
        uint32_t tid = threadIdx.x;
        int32_t _temp = index;
        int32_t const cCol = _temp % channels;
        _temp /= channels;
        int32_t const samplingIndex = _temp;
        int32_t const mCol = _temp % numHeads;
        _temp /= numHeads;
        int32_t const qCol = _temp % numQuery;
        _temp /= numQuery;
        int32_t const bCol = _temp;

        scalar_t const topGrad = grad_col[index];

        int32_t dataWeightPtr = samplingIndex * numLevels * numPoint;
        int32_t dataLocWPtr = dataWeightPtr << 1;
        int32_t const gradSamplingPtr = dataWeightPtr;
        gradSamplingLoc += gradSamplingPtr << 1;
        gradAttnWeight += gradSamplingPtr;
        int32_t const gradWeightStride = 1;
        int32_t const gradLocStride = 2;
        int32_t const qidStride = numHeads * channels;
        int32_t const dataValuePtrInitOffset = bCol * spatialSize * qidStride;

        for (int32_t lCol = 0; lCol < numLevels; ++lCol)
        {
            int32_t const levelStartId = dataLevelStartIndex[lCol];
            int32_t const spatialHPtr = lCol << 1;
            int32_t const spatialH = dataSpatialShapes[spatialHPtr];
            int32_t const spatialW = dataSpatialShapes[spatialHPtr + 1];
            int32_t const valuePtrOffset = dataValuePtrInitOffset + levelStartId * qidStride;
            scalar_t const* dataValuePtr = dataValue + valuePtrOffset;
            scalar_t* gradValuePtr = gradValue + valuePtrOffset;

            for (int32_t pCol = 0; pCol < numPoint; ++pCol)
            {
                scalar_t const locW = dataSamplingLoc[dataLocWPtr];
                scalar_t const locH = dataSamplingLoc[dataLocWPtr + 1];
                scalar_t const weight = dataAttnWeight[dataWeightPtr];

                scalar_t const hIm = locH * spatialH - 0.5;
                scalar_t const wIm = locW * spatialW - 0.5;
                *(cacheGradSamplingLoc + (threadIdx.x << 1)) = 0;
                *(cacheGradSamplingLoc + ((threadIdx.x << 1) + 1)) = 0;
                *(cacheGradAttnWeight + threadIdx.x) = 0;
                if (hIm > -1 && wIm > -1 && hIm < spatialH && wIm < spatialW)
                {
                    ms_deform_attn_col2im_bilinear(dataValuePtr, spatialH, spatialW, numHeads, channels, hIm, wIm, mCol,
                        cCol, topGrad, weight, gradValuePtr, cacheGradSamplingLoc + (threadIdx.x << 1),
                        cacheGradAttnWeight + threadIdx.x);
                }

                __syncthreads();
                if (tid == 0)
                {
                    scalar_t _gradW = cacheGradSamplingLoc[0], _gradH = cacheGradSamplingLoc[1],
                             _gradA = cacheGradAttnWeight[0];
                    int32_t sid = 2;
                    for (uint32_t tid = 1; tid < blockDim.x; ++tid)
                    {
                        _gradW += cacheGradSamplingLoc[sid];
                        _gradH += cacheGradSamplingLoc[sid + 1];
                        _gradA += cacheGradAttnWeight[tid];
                        sid += 2;
                    }

                    *gradSamplingLoc = _gradW;
                    *(gradSamplingLoc + 1) = _gradH;
                    *gradAttnWeight = _gradA;
                }
                __syncthreads();

                dataWeightPtr += 1;
                dataLocWPtr += 2;
                gradAttnWeight += gradWeightStride;
                gradSamplingLoc += gradLocStride;
            }
        }
    }
}

template <typename scalar_t>
__global__ void ms_deformable_col2im_gpu_kernel_shm_reduce_v2(int32_t const n, scalar_t const* grad_col,
    scalar_t const* dataValue, int32_t const* dataSpatialShapes, int32_t const* dataLevelStartIndex,
    scalar_t const* dataSamplingLoc, scalar_t const* dataAttnWeight, int32_t const batchSize, int32_t const spatialSize,
    int32_t const numHeads, int32_t const channels, int32_t const numLevels, int32_t const numQuery, int32_t const numPoint,
    scalar_t* gradValue, scalar_t* gradSamplingLoc, scalar_t* gradAttnWeight)
{
    CUDA_KERNEL_LOOP(index, n)
    {
        extern __shared__ int32_t _s[];
        scalar_t* cacheGradSamplingLoc = (scalar_t*) _s;
        scalar_t* cacheGradAttnWeight = cacheGradSamplingLoc + 2 * blockDim.x;
        uint32_t tid = threadIdx.x;
        int32_t _temp = index;
        int32_t const cCol = _temp % channels;
        _temp /= channels;
        int32_t const samplingIndex = _temp;
        int32_t const mCol = _temp % numHeads;
        _temp /= numHeads;
        int32_t const qCol = _temp % numQuery;
        _temp /= numQuery;
        int32_t const bCol = _temp;

        scalar_t const topGrad = grad_col[index];

        int32_t dataWeightPtr = samplingIndex * numLevels * numPoint;
        int32_t dataLocWPtr = dataWeightPtr << 1;
        int32_t const gradSamplingPtr = dataWeightPtr;
        gradSamplingLoc += gradSamplingPtr << 1;
        gradAttnWeight += gradSamplingPtr;
        int32_t const gradWeightStride = 1;
        int32_t const gradLocStride = 2;
        int32_t const qidStride = numHeads * channels;
        int32_t const dataValuePtrInitOffset = bCol * spatialSize * qidStride;

        for (int32_t lCol = 0; lCol < numLevels; ++lCol)
        {
            int32_t const levelStartId = dataLevelStartIndex[lCol];
            int32_t const spatialHPtr = lCol << 1;
            int32_t const spatialH = dataSpatialShapes[spatialHPtr];
            int32_t const spatialW = dataSpatialShapes[spatialHPtr + 1];
            int32_t const valuePtrOffset = dataValuePtrInitOffset + levelStartId * qidStride;
            scalar_t const* dataValuePtr = dataValue + valuePtrOffset;
            scalar_t* gradValuePtr = gradValue + valuePtrOffset;

            for (int32_t pCol = 0; pCol < numPoint; ++pCol)
            {
                scalar_t const locW = dataSamplingLoc[dataLocWPtr];
                scalar_t const locH = dataSamplingLoc[dataLocWPtr + 1];
                scalar_t const weight = dataAttnWeight[dataWeightPtr];

                scalar_t const hIm = locH * spatialH - 0.5;
                scalar_t const wIm = locW * spatialW - 0.5;
                *(cacheGradSamplingLoc + (threadIdx.x << 1)) = 0;
                *(cacheGradSamplingLoc + ((threadIdx.x << 1) + 1)) = 0;
                *(cacheGradAttnWeight + threadIdx.x) = 0;
                if (hIm > -1 && wIm > -1 && hIm < spatialH && wIm < spatialW)
                {
                    ms_deform_attn_col2im_bilinear(dataValuePtr, spatialH, spatialW, numHeads, channels, hIm, wIm, mCol,
                        cCol, topGrad, weight, gradValuePtr, cacheGradSamplingLoc + (threadIdx.x << 1),
                        cacheGradAttnWeight + threadIdx.x);
                }

                __syncthreads();

                for (uint32_t s = blockDim.x / 2, spre = blockDim.x; s > 0; s >>= 1, spre >>= 1)
                {
                    if (tid < s)
                    {
                        uint32_t const xid1 = tid << 1;
                        uint32_t const xid2 = (tid + s) << 1;
                        cacheGradAttnWeight[tid] += cacheGradAttnWeight[tid + s];
                        cacheGradSamplingLoc[xid1] += cacheGradSamplingLoc[xid2];
                        cacheGradSamplingLoc[xid1 + 1] += cacheGradSamplingLoc[xid2 + 1];
                        if (tid + (s << 1) < spre)
                        {
                            cacheGradAttnWeight[tid] += cacheGradAttnWeight[tid + (s << 1)];
                            cacheGradSamplingLoc[xid1] += cacheGradSamplingLoc[xid2 + (s << 1)];
                            cacheGradSamplingLoc[xid1 + 1] += cacheGradSamplingLoc[xid2 + 1 + (s << 1)];
                        }
                    }
                    __syncthreads();
                }

                if (tid == 0)
                {
                    *gradSamplingLoc = cacheGradSamplingLoc[0];
                    *(gradSamplingLoc + 1) = cacheGradSamplingLoc[1];
                    *gradAttnWeight = cacheGradAttnWeight[0];
                }
                __syncthreads();

                dataWeightPtr += 1;
                dataLocWPtr += 2;
                gradAttnWeight += gradWeightStride;
                gradSamplingLoc += gradLocStride;
            }
        }
    }
}

template <typename scalar_t>
__global__ void ms_deformable_col2im_gpu_kernel_shm_reduce_v2_multi_blocks(int32_t const n, scalar_t const* grad_col,
    scalar_t const* dataValue, int32_t const* dataSpatialShapes, int32_t const* dataLevelStartIndex,
    scalar_t const* dataSamplingLoc, scalar_t const* dataAttnWeight, int32_t const batchSize, int32_t const spatialSize,
    int32_t const numHeads, int32_t const channels, int32_t const numLevels, int32_t const numQuery, int32_t const numPoint,
    scalar_t* gradValue, scalar_t* gradSamplingLoc, scalar_t* gradAttnWeight)
{
    CUDA_KERNEL_LOOP(index, n)
    {
        extern __shared__ int32_t _s[];
        scalar_t* cacheGradSamplingLoc = (scalar_t*) _s;
        scalar_t* cacheGradAttnWeight = cacheGradSamplingLoc + 2 * blockDim.x;
        uint32_t tid = threadIdx.x;
        int32_t _temp = index;
        int32_t const cCol = _temp % channels;
        _temp /= channels;
        int32_t const samplingIndex = _temp;
        int32_t const mCol = _temp % numHeads;
        _temp /= numHeads;
        int32_t const qCol = _temp % numQuery;
        _temp /= numQuery;
        int32_t const bCol = _temp;

        scalar_t const topGrad = grad_col[index];

        int32_t dataWeightPtr = samplingIndex * numLevels * numPoint;
        int32_t dataLocWPtr = dataWeightPtr << 1;
        int32_t const gradSamplingPtr = dataWeightPtr;
        gradSamplingLoc += gradSamplingPtr << 1;
        gradAttnWeight += gradSamplingPtr;
        int32_t const gradWeightStride = 1;
        int32_t const gradLocStride = 2;
        int32_t const qidStride = numHeads * channels;
        int32_t const dataValuePtrInitOffset = bCol * spatialSize * qidStride;

        for (int32_t lCol = 0; lCol < numLevels; ++lCol)
        {
            int32_t const levelStartId = dataLevelStartIndex[lCol];
            int32_t const spatialHPtr = lCol << 1;
            int32_t const spatialH = dataSpatialShapes[spatialHPtr];
            int32_t const spatialW = dataSpatialShapes[spatialHPtr + 1];
            int32_t const valuePtrOffset = dataValuePtrInitOffset + levelStartId * qidStride;
            scalar_t const* dataValuePtr = dataValue + valuePtrOffset;
            scalar_t* gradValuePtr = gradValue + valuePtrOffset;

            for (int32_t pCol = 0; pCol < numPoint; ++pCol)
            {
                scalar_t const locW = dataSamplingLoc[dataLocWPtr];
                scalar_t const locH = dataSamplingLoc[dataLocWPtr + 1];
                scalar_t const weight = dataAttnWeight[dataWeightPtr];

                scalar_t const hIm = locH * spatialH - 0.5;
                scalar_t const wIm = locW * spatialW - 0.5;
                *(cacheGradSamplingLoc + (threadIdx.x << 1)) = 0;
                *(cacheGradSamplingLoc + ((threadIdx.x << 1) + 1)) = 0;
                *(cacheGradAttnWeight + threadIdx.x) = 0;
                if (hIm > -1 && wIm > -1 && hIm < spatialH && wIm < spatialW)
                {
                    ms_deform_attn_col2im_bilinear(dataValuePtr, spatialH, spatialW, numHeads, channels, hIm, wIm, mCol,
                        cCol, topGrad, weight, gradValuePtr, cacheGradSamplingLoc + (threadIdx.x << 1),
                        cacheGradAttnWeight + threadIdx.x);
                }

                __syncthreads();

                for (uint32_t s = blockDim.x / 2, spre = blockDim.x; s > 0; s >>= 1, spre >>= 1)
                {
                    if (tid < s)
                    {
                        uint32_t const xid1 = tid << 1;
                        uint32_t const xid2 = (tid + s) << 1;
                        cacheGradAttnWeight[tid] += cacheGradAttnWeight[tid + s];
                        cacheGradSamplingLoc[xid1] += cacheGradSamplingLoc[xid2];
                        cacheGradSamplingLoc[xid1 + 1] += cacheGradSamplingLoc[xid2 + 1];
                        if (tid + (s << 1) < spre)
                        {
                            cacheGradAttnWeight[tid] += cacheGradAttnWeight[tid + (s << 1)];
                            cacheGradSamplingLoc[xid1] += cacheGradSamplingLoc[xid2 + (s << 1)];
                            cacheGradSamplingLoc[xid1 + 1] += cacheGradSamplingLoc[xid2 + 1 + (s << 1)];
                        }
                    }
                    __syncthreads();
                }

                if (tid == 0)
                {
                    atomicAdd(gradSamplingLoc, cacheGradSamplingLoc[0]);
                    atomicAdd(gradSamplingLoc + 1, cacheGradSamplingLoc[1]);
                    atomicAdd(gradAttnWeight, cacheGradAttnWeight[0]);
                }
                __syncthreads();

                dataWeightPtr += 1;
                dataLocWPtr += 2;
                gradAttnWeight += gradWeightStride;
                gradSamplingLoc += gradLocStride;
            }
        }
    }
}

template <typename scalar_t>
__global__ void ms_deformable_col2im_gpu_kernel_gm(int32_t const n, scalar_t const* grad_col, scalar_t const* dataValue,
    int32_t const* dataSpatialShapes, int32_t const* dataLevelStartIndex, scalar_t const* dataSamplingLoc,
    scalar_t const* dataAttnWeight, int32_t const batchSize, int32_t const spatialSize, int32_t const numHeads, int32_t const channels,
    int32_t const numLevels, int32_t const numQuery, int32_t const numPoint, scalar_t* gradValue, scalar_t* gradSamplingLoc,
    scalar_t* gradAttnWeight)
{
    CUDA_KERNEL_LOOP(index, n)
    {
        int32_t _temp = index;
        int32_t const cCol = _temp % channels;
        _temp /= channels;
        int32_t const samplingIndex = _temp;
        int32_t const mCol = _temp % numHeads;
        _temp /= numHeads;
        int32_t const qCol = _temp % numQuery;
        _temp /= numQuery;
        int32_t const bCol = _temp;

        scalar_t const topGrad = grad_col[index];

        int32_t dataWeightPtr = samplingIndex * numLevels * numPoint;
        int32_t dataLocWPtr = dataWeightPtr << 1;
        int32_t const gradSamplingPtr = dataWeightPtr;
        gradSamplingLoc += gradSamplingPtr << 1;
        gradAttnWeight += gradSamplingPtr;
        int32_t const gradWeightStride = 1;
        int32_t const gradLocStride = 2;
        int32_t const qidStride = numHeads * channels;
        int32_t const dataValuePtrInitOffset = bCol * spatialSize * qidStride;

        for (int32_t lCol = 0; lCol < numLevels; ++lCol)
        {
            int32_t const levelStartId = dataLevelStartIndex[lCol];
            int32_t const spatialHPtr = lCol << 1;
            int32_t const spatialH = dataSpatialShapes[spatialHPtr];
            int32_t const spatialW = dataSpatialShapes[spatialHPtr + 1];
            int32_t const valuePtrOffset = dataValuePtrInitOffset + levelStartId * qidStride;
            scalar_t const* dataValuePtr = dataValue + valuePtrOffset;
            scalar_t* gradValuePtr = gradValue + valuePtrOffset;

            for (int32_t pCol = 0; pCol < numPoint; ++pCol)
            {
                scalar_t const locW = dataSamplingLoc[dataLocWPtr];
                scalar_t const locH = dataSamplingLoc[dataLocWPtr + 1];
                scalar_t const weight = dataAttnWeight[dataWeightPtr];

                scalar_t const hIm = locH * spatialH - 0.5;
                scalar_t const wIm = locW * spatialW - 0.5;
                if (hIm > -1 && wIm > -1 && hIm < spatialH && wIm < spatialW)
                {
                    ms_deform_attn_col2im_bilinear_gm(dataValuePtr, spatialH, spatialW, numHeads, channels, hIm, wIm,
                        mCol, cCol, topGrad, weight, gradValuePtr, gradSamplingLoc, gradAttnWeight);
                }
                dataWeightPtr += 1;
                dataLocWPtr += 2;
                gradAttnWeight += gradWeightStride;
                gradSamplingLoc += gradLocStride;
            }
        }
    }
}

template <typename scalar_t>
void ms_deformable_im2col_cuda(cudaStream_t stream, scalar_t const* dataValue, int32_t const* dataSpatialShapes,
    int32_t const* dataLevelStartIndex, scalar_t const* dataSamplingLoc, scalar_t const* dataAttnWeight,
    int32_t const batchSize, int32_t const spatialSize, int32_t const numHeads, int32_t const channels, int32_t const numLevels,
    int32_t const numQuery, int32_t const numPoint, scalar_t* dataCol)
{
    int32_t const numKernels = batchSize * numQuery * numHeads * channels;
    int32_t const numActualKernels = batchSize * numQuery * numHeads * channels;
    int32_t const numThreads = kCUDA_NUM_THREADS;
    cudaError_t err = cudaSuccess;

    ms_deformable_im2col_gpu_kernel<scalar_t><<<GET_BLOCKS(numActualKernels, numThreads), numThreads, 0, stream>>>(
        numKernels, dataValue, dataSpatialShapes, dataLevelStartIndex, dataSamplingLoc, dataAttnWeight, batchSize,
        spatialSize, numHeads, channels, numLevels, numQuery, numPoint, dataCol);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        // nvinfer1::plugin::gLogError << "error in ms_deformable_im2col_cuda: " << cudaGetErrorString(err) << std::endl;
    }
}

// ========== 模板特化：__half 走 half2；奇数通道回退 ==========
template <>
inline void ms_deformable_im2col_cuda<__half>(
    cudaStream_t stream,
    const __half* dataValue, const int32_t* dataSpatialShapes,
    const int32_t* dataLevelStartIndex, const __half* dataSamplingLoc,
    const __half* dataAttnWeight, int32_t batchSize, int32_t spatialSize,
    int32_t numHeads, int32_t channels, int32_t numLevels,
    int32_t numQuery, int32_t numPoint, __half* dataCol)
{
    // if ((channels & 1) == 0) {
    if (1) {
        ms_deformable_im2col_cuda_half2(stream,
            dataValue, dataSpatialShapes, dataLevelStartIndex,
            dataSamplingLoc, dataAttnWeight,
            batchSize, spatialSize, numHeads, channels,
            numLevels, numQuery, numPoint, dataCol);
    } else {
        // 奇数通道：回退到原 half scalar 版本
        int32_t const numKernels = batchSize * numQuery * numHeads * channels ;
        int32_t const numThreads = kCUDA_NUM_THREADS;
        ms_deformable_im2col_gpu_kernel<__half>
            <<<GET_BLOCKS(numKernels, numThreads), numThreads, 0, stream>>>(
                numKernels, dataValue, dataSpatialShapes, dataLevelStartIndex,
                dataSamplingLoc, dataAttnWeight, batchSize, spatialSize,
                numHeads, channels, numLevels, numQuery, numPoint, dataCol);
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        // nvinfer1::plugin::gLogError << "ms_deformable_im2col_cuda<half>: "
                                    // << cudaGetErrorString(err) <<std::endl;
    }
}

template <typename scalar_t>
void ms_deformable_col2im_cuda(cudaStream_t stream, scalar_t const* grad_col, scalar_t const* dataValue,
    int32_t const* dataSpatialShapes, int32_t const* dataLevelStartIndex, scalar_t const* dataSamplingLoc,
    scalar_t const* dataAttnWeight, int32_t const batchSize, int32_t const spatialSize, int32_t const numHeads, int32_t const channels,
    int32_t const numLevels, int32_t const numQuery, int32_t const numPoint, scalar_t* gradValue, scalar_t* gradSamplingLoc,
    scalar_t* gradAttnWeight)
{
    int32_t const numThreads = (channels > kCUDA_NUM_THREADS) ? kCUDA_NUM_THREADS : channels;
    int32_t const numKernels = batchSize * numQuery * numHeads * channels;
    int32_t const numActualKernels = batchSize * numQuery * numHeads * channels;
    if (channels > 1024)
    {
        if ((channels & 1023) == 0)
        {
            ms_deformable_col2im_gpu_kernel_shm_reduce_v2_multi_blocks<scalar_t>
                <<<GET_BLOCKS(numActualKernels, numThreads), numThreads, numThreads * 3 * sizeof(scalar_t), stream>>>(
                    numKernels, grad_col, dataValue, dataSpatialShapes, dataLevelStartIndex, dataSamplingLoc,
                    dataAttnWeight, batchSize, spatialSize, numHeads, channels, numLevels, numQuery, numPoint,
                    gradValue, gradSamplingLoc, gradAttnWeight);
        }
        else
        {
            ms_deformable_col2im_gpu_kernel_gm<scalar_t>
                <<<GET_BLOCKS(numActualKernels, numThreads), numThreads, 0, stream>>>(numKernels, grad_col, dataValue,
                    dataSpatialShapes, dataLevelStartIndex, dataSamplingLoc, dataAttnWeight, batchSize, spatialSize,
                    numHeads, channels, numLevels, numQuery, numPoint, gradValue, gradSamplingLoc, gradAttnWeight);
        }
    }
    else
    {
        switch (channels)
        {
        case 1:
            ms_deformable_col2im_gpu_kernel_shm_blocksize_aware_reduce_v1<scalar_t, 1>
                <<<GET_BLOCKS(numActualKernels, numThreads), numThreads, 0, stream>>>(numKernels, grad_col, dataValue,
                    dataSpatialShapes, dataLevelStartIndex, dataSamplingLoc, dataAttnWeight, batchSize, spatialSize,
                    numHeads, channels, numLevels, numQuery, numPoint, gradValue, gradSamplingLoc, gradAttnWeight);
            break;
        case 2:
            ms_deformable_col2im_gpu_kernel_shm_blocksize_aware_reduce_v1<scalar_t, 2>
                <<<GET_BLOCKS(numActualKernels, numThreads), numThreads, 0, stream>>>(numKernels, grad_col, dataValue,
                    dataSpatialShapes, dataLevelStartIndex, dataSamplingLoc, dataAttnWeight, batchSize, spatialSize,
                    numHeads, channels, numLevels, numQuery, numPoint, gradValue, gradSamplingLoc, gradAttnWeight);
            break;
        case 4:
            ms_deformable_col2im_gpu_kernel_shm_blocksize_aware_reduce_v1<scalar_t, 4>
                <<<GET_BLOCKS(numActualKernels, numThreads), numThreads, 0, stream>>>(numKernels, grad_col, dataValue,
                    dataSpatialShapes, dataLevelStartIndex, dataSamplingLoc, dataAttnWeight, batchSize, spatialSize,
                    numHeads, channels, numLevels, numQuery, numPoint, gradValue, gradSamplingLoc, gradAttnWeight);
            break;
        case 8:
            ms_deformable_col2im_gpu_kernel_shm_blocksize_aware_reduce_v1<scalar_t, 8>
                <<<GET_BLOCKS(numActualKernels, numThreads), numThreads, 0, stream>>>(numKernels, grad_col, dataValue,
                    dataSpatialShapes, dataLevelStartIndex, dataSamplingLoc, dataAttnWeight, batchSize, spatialSize,
                    numHeads, channels, numLevels, numQuery, numPoint, gradValue, gradSamplingLoc, gradAttnWeight);
            break;
        case 16:
            ms_deformable_col2im_gpu_kernel_shm_blocksize_aware_reduce_v1<scalar_t, 16>
                <<<GET_BLOCKS(numActualKernels, numThreads), numThreads, 0, stream>>>(numKernels, grad_col, dataValue,
                    dataSpatialShapes, dataLevelStartIndex, dataSamplingLoc, dataAttnWeight, batchSize, spatialSize,
                    numHeads, channels, numLevels, numQuery, numPoint, gradValue, gradSamplingLoc, gradAttnWeight);
            break;
        case 32:
            ms_deformable_col2im_gpu_kernel_shm_blocksize_aware_reduce_v1<scalar_t, 32>
                <<<GET_BLOCKS(numActualKernels, numThreads), numThreads, 0, stream>>>(numKernels, grad_col, dataValue,
                    dataSpatialShapes, dataLevelStartIndex, dataSamplingLoc, dataAttnWeight, batchSize, spatialSize,
                    numHeads, channels, numLevels, numQuery, numPoint, gradValue, gradSamplingLoc, gradAttnWeight);
            break;
        case 64:
            ms_deformable_col2im_gpu_kernel_shm_blocksize_aware_reduce_v2<scalar_t, 64>
                <<<GET_BLOCKS(numActualKernels, numThreads), numThreads, 0, stream>>>(numKernels, grad_col, dataValue,
                    dataSpatialShapes, dataLevelStartIndex, dataSamplingLoc, dataAttnWeight, batchSize, spatialSize,
                    numHeads, channels, numLevels, numQuery, numPoint, gradValue, gradSamplingLoc, gradAttnWeight);
            break;
        case 128:
            ms_deformable_col2im_gpu_kernel_shm_blocksize_aware_reduce_v2<scalar_t, 128>
                <<<GET_BLOCKS(numActualKernels, numThreads), numThreads, 0, stream>>>(numKernels, grad_col, dataValue,
                    dataSpatialShapes, dataLevelStartIndex, dataSamplingLoc, dataAttnWeight, batchSize, spatialSize,
                    numHeads, channels, numLevels, numQuery, numPoint, gradValue, gradSamplingLoc, gradAttnWeight);
            break;
        case 256:
            ms_deformable_col2im_gpu_kernel_shm_blocksize_aware_reduce_v2<scalar_t, 256>
                <<<GET_BLOCKS(numActualKernels, numThreads), numThreads, 0, stream>>>(numKernels, grad_col, dataValue,
                    dataSpatialShapes, dataLevelStartIndex, dataSamplingLoc, dataAttnWeight, batchSize, spatialSize,
                    numHeads, channels, numLevels, numQuery, numPoint, gradValue, gradSamplingLoc, gradAttnWeight);
            break;
        case 512:
            ms_deformable_col2im_gpu_kernel_shm_blocksize_aware_reduce_v2<scalar_t, 512>
                <<<GET_BLOCKS(numActualKernels, numThreads), numThreads, 0, stream>>>(numKernels, grad_col, dataValue,
                    dataSpatialShapes, dataLevelStartIndex, dataSamplingLoc, dataAttnWeight, batchSize, spatialSize,
                    numHeads, channels, numLevels, numQuery, numPoint, gradValue, gradSamplingLoc, gradAttnWeight);
            break;
        case 1024:
            ms_deformable_col2im_gpu_kernel_shm_blocksize_aware_reduce_v2<scalar_t, 1024>
                <<<GET_BLOCKS(numActualKernels, numThreads), numThreads, 0, stream>>>(numKernels, grad_col, dataValue,
                    dataSpatialShapes, dataLevelStartIndex, dataSamplingLoc, dataAttnWeight, batchSize, spatialSize,
                    numHeads, channels, numLevels, numQuery, numPoint, gradValue, gradSamplingLoc, gradAttnWeight);
            break;
        default:
            if (channels < 64)
            {
                ms_deformable_col2im_gpu_kernel_shm_reduce_v1<scalar_t><<<GET_BLOCKS(numActualKernels, numThreads),
                    numThreads, numThreads * 3 * sizeof(scalar_t), stream>>>(numKernels, grad_col, dataValue,
                    dataSpatialShapes, dataLevelStartIndex, dataSamplingLoc, dataAttnWeight, batchSize, spatialSize,
                    numHeads, channels, numLevels, numQuery, numPoint, gradValue, gradSamplingLoc, gradAttnWeight);
            }
            else
            {
                ms_deformable_col2im_gpu_kernel_shm_reduce_v2<scalar_t><<<GET_BLOCKS(numActualKernels, numThreads),
                    numThreads, numThreads * 3 * sizeof(scalar_t), stream>>>(numKernels, grad_col, dataValue,
                    dataSpatialShapes, dataLevelStartIndex, dataSamplingLoc, dataAttnWeight, batchSize, spatialSize,
                    numHeads, channels, numLevels, numQuery, numPoint, gradValue, gradSamplingLoc, gradAttnWeight);
            }
        }
    }
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        // nvinfer1::plugin::gLogError << "error in ms_deformable_col2im_cuda: " << cudaGetErrorString(err) << std::endl;
    }
}

#define CUDA_KERNEL_LOOP_RANGE(tid, nDataMin, nDataMax)                                                                \
    for (int32_t tid = blockIdx.x * blockDim.x + threadIdx.x; ((tid >= (nDataMin)) && (tid < (nDataMax)));                 \
         tid += blockDim.x * gridDim.x)

__global__ void float2half_input(int32_t const nData1, int32_t const nData2, int32_t const nData3, float const* data1Float,
    float const* data2Float, float const* data3Float, __half* data1Half, __half* data2Half, __half* data3Half)
{
    CUDA_KERNEL_LOOP(index, nData1)
    {
        data1Half[index] = __float2half(data1Float[index]);
        data2Half[index] = __float2half(data2Float[index]);
        data3Half[index] = __float2half(data3Float[index]);
    }

    CUDA_KERNEL_LOOP_RANGE(index, nData1, nData2)
    {
        data2Half[index] = __float2half(data2Float[index]);
        data3Half[index] = __float2half(data3Float[index]);
    }

    CUDA_KERNEL_LOOP_RANGE(index, nData2, nData3)
    {
        data3Half[index] = __float2half(data3Float[index]);
    }
}

__global__ void half2float_output(int32_t const n_data, __half const* data_half, float* data_float)
{
    CUDA_KERNEL_LOOP(index, n_data)
    {
        data_float[index] = __half2float(data_half[index]);
    }
}

#endif
