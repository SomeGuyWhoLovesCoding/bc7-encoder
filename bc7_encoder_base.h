#ifndef BC7_ENCODER_BASE_H
#define BC7_ENCODER_BASE_H

// bc7_encoder_base.h - BC7 encoder (converted from bc7e.ispc)
// Original: bc7e.ispc - Fast high quality SIMD BC7 encoder
// Copyright (C) 2018-2020 Binomial LLC, All rights reserved.
// Apache 2.0 license - see LICENSE.
//
// NOTE: This is a scalar C++ fallback port from the original ISPC code.
//       The original ISPC code uses SIMD (varying) types for parallel block encoding.
//       This port replaces ISPC "varying" types with scalar C++ equivalents.
//       For maximum performance, use the original ISPC-compiled version.

#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <algorithm>
#include <cmath>
#include <climits>
#include <cassert>

// SSE2 acceleration for the hottest error-metric inner loops.
// Guarded so the encoder still builds on targets without SSE2.
#if defined(__SSE2__) || defined(_M_X64) || defined(_M_AMD64)
    #include <emmintrin.h>
    #define BC7E_HAVE_SSE2 1
#else
    #define BC7E_HAVE_SSE2 0
#endif

#define ISPC_NO_XEON_PHI 1

using std::min;
using std::max;

// ISPC builtins ported to scalar C++
#ifndef ispc_clamp
#define ispc_clamp(x, lo, hi) (((x) < (lo)) ? (lo) : (((x) > (hi)) ? (hi) : (x)))
#endif

template<typename T>
static inline T ispc_select(bool cond, T a, T b) { return cond ? a : b; }

// bc7e.ispc - Fast high quality SIMD BC7 encoder
// Copyright (C) 2018-2020 Binomial LLC, All rights reserved. Apache 2.0 license - see LICENSE.
// Typically compiled as: ispc -g -O2 "%(Filename).ispc" -o "$(TargetDir)%(Filename).obj" -h "$(ProjectDir)%(Filename)_ispc.h" --target=sse2,sse4,avx,avx2 --opt=fast-math --opt=disable-assertions
// --opt=fast-math is optional (doesn't make much if any measurable difference). 

#define BC7E_2SUBSET_CHECKERBOARD_PARTITION_INDEX (34)
#define BC7E_BLOCK_SIZE (16)
#define BC7E_MAX_PARTITIONS0 (8)
#define BC7E_MAX_PARTITIONS1 (16) // set from 32 to 16
#define BC7E_MAX_PARTITIONS2 (16) // keep this at 16 minimum
#define BC7E_MAX_PARTITIONS3 (32) // was 32 but i dont think mode 3 uhhhhhhh
#define BC7E_MAX_PARTITIONS7 (32)
#define BC7E_MAX_UBER_LEVEL (4)

// typedef unsigned int8 uint8_t;  [replaced by <cstdint>]
// typedef unsigned int64_t uint16_t;  [replaced by <cstdint>]
// typedef unsigned int32 uint32_t;  [replaced by <cstdint>]
// typedef unsigned int64_t uint64_t;  [replaced by <cstdint>]
// typedef int8 int8_t;  [replaced by <cstdint>]
// typedef int int32_t;  [replaced by <cstdint>]
// typedef int64_t int64_t;  [replaced by <cstdint>]

#ifndef UINT16_MAX
#define UINT16_MAX (0xFFFF)
#endif

#ifndef UINT_MAX
#define UINT_MAX (0xFFFFFFFFU)
#endif

#ifndef UINT64_MAX
#define UINT64_MAX (0xFFFFFFFFFFFFFFFFULL)
#endif

#ifndef INT64_MAX
#define INT64_MAX (0x7FFFFFFFFFFFFFFFULL)
#endif

struct bc7e_compress_block_params
{
                uint32_t m_max_partitions_mode[8];

                uint32_t m_weights[4];

                uint32_t m_uber_level;
                uint32_t m_refinement_passes;
                
                uint32_t m_mode4_rotation_mask;
                uint32_t m_mode4_index_mask;
                uint32_t m_mode5_rotation_mask;
                uint32_t m_uber1_mask;
                
                bool m_perceptual;
                bool m_pbit_search;
                bool m_mode6_only;
                bool m_unused0;
                
                struct
                {
                                uint32_t m_max_mode13_partitions_to_try;
                                uint32_t m_max_mode0_partitions_to_try;
                                uint32_t m_max_mode2_partitions_to_try;
                                bool m_use_mode[8]; // FIX A: was [7] but indexed up to [7] in init -> OOB write
                                bool m_unused1;
                } m_opaque_settings;

                struct
                {
                                uint32_t m_max_mode7_partitions_to_try;
                                uint32_t m_mode67_error_weight_mul[4];
                                                                
                                bool m_use_mode4;
                                bool m_use_mode5;
                                bool m_use_mode6;
                                bool m_use_mode7;

                                bool m_use_mode4_rotation;
                                bool m_use_mode5_rotation;
                                bool m_unused2;
                                bool m_unused3;
                } m_alpha_settings;

};

static inline int32_t clampi(int32_t value, int32_t low, int32_t high) { return ispc_clamp(value, low, high); }
static inline uint32_t clampu(uint32_t value, uint32_t low, uint32_t high) { return (std::min)((std::max)(value, low), high); }
static inline float clampf(float value, float low, float high) { return ispc_clamp(value, low, high); }

static inline float saturate(float value) { return clampf(value, 0, 1.0f); }
static inline float saturate255(float value) { return clampf(value, 0, 255.0f); }

static inline uint8_t minimumub(uint8_t a, uint8_t b) { return (std::min)(a, b); }
static inline int32_t minimumi(int32_t a, int32_t b) { return (std::min)(a, b); }
static inline uint32_t minimumu(uint32_t a, uint32_t b) { return (std::min)(a, b); }
static inline uint64_t minimumu64(uint64_t a, uint64_t b) { return (std::min)(a, b); }
static inline float minimumf(float a, float b) { return (std::min)(a, b); }

static inline uint8_t maximumub(uint8_t a, uint8_t b) { return (std::max)(a, b); }
static inline int32_t maximumi(int32_t a, int32_t b) { return (std::max)(a, b); }
static inline uint32_t maximumu(uint32_t a, uint32_t b) { return (std::max)(a, b); }
static inline float maximumf(float a, float b) { return (std::max)(a, b); }

static inline int32_t iabs32(int32_t v) { uint32_t msk = (uint32_t)(v >> 31); return (v ^ (int32_t)msk) - (int32_t)msk; }

static inline void swapub(uint8_t* a, uint8_t* b) { uint8_t t = *a; *a = *b; *b = t; }
static inline void swapi(int32_t* a, int32_t* b) { int32_t t = *a; *a = *b; *b = t; }
static inline void swapu(uint32_t* a, uint32_t* b) { uint32_t t = *a; *a = *b; *b = t; }
static inline void swapf(float* a, float* b) { float t = *a; *a = *b; *b = t; }

static inline float square(float s) { return s * s; }
static inline int square(int s) { return s * s; }

struct color_quad_u8
{
                uint8_t m_c[4];
};

struct color_quad_i
{
                int32_t m_c[4];
};

struct color_quad_f
{
                float m_c[4];
};

static inline color_quad_i component_min_rgb(const  color_quad_i *  pA, const  color_quad_i *  pB) 
{ 
                color_quad_i res; 
                res.m_c[0] = minimumi(pA->m_c[0], pB->m_c[0]);
                res.m_c[1] = minimumi(pA->m_c[1], pB->m_c[1]);
                res.m_c[2] = minimumi(pA->m_c[2], pB->m_c[2]);
                res.m_c[3] = 255;
                return res;
}

static inline color_quad_i component_max_rgb(const  color_quad_i *  pA, const  color_quad_i *  pB) 
{ 
                color_quad_i res;
                res.m_c[0] = maximumi(pA->m_c[0], pB->m_c[0]);
                res.m_c[1] = maximumi(pA->m_c[1], pB->m_c[1]);
                res.m_c[2] = maximumi(pA->m_c[2], pB->m_c[2]);
                res.m_c[3] = 255;
                return res;
}

static inline  color_quad_i *color_quad_i_set_clamped( color_quad_i *  pRes,  int32_t r,  int32_t g,  int32_t b,  int32_t a)
{
                pRes->m_c[0] = clampi(r, 0, 255);
                pRes->m_c[1] = clampi(g, 0, 255);
                pRes->m_c[2] = clampi(b, 0, 255);
                pRes->m_c[3] = clampi(a, 0, 255);
                return pRes;
}

static inline  color_quad_i *color_quad_i_set( color_quad_i *  pRes,  int32_t r,  int32_t g,  int32_t b,  int32_t a)
{
                pRes->m_c[0] = r;
                pRes->m_c[1] = g;
                pRes->m_c[2] = b;
                pRes->m_c[3] = a;
                return pRes;
}


static inline bool color_quad_i_equals(const  color_quad_i *  pLHS, const  color_quad_i *  pRHS)
{
                return (pLHS->m_c[0] == pRHS->m_c[0]) && (pLHS->m_c[1] == pRHS->m_c[1]) && (pLHS->m_c[2] == pRHS->m_c[2]) && (pLHS->m_c[3] == pRHS->m_c[3]);
}

static inline bool color_quad_i_notequals(const  color_quad_i *  pLHS, const  color_quad_i *  pRHS)
{
                return !color_quad_i_equals(pLHS, pRHS);
}

struct vec4F
{
                float m_c[4];
};

static inline  vec4F *  vec4F_set_scalar( vec4F *  pV, float x)
{
                pV->m_c[0] = x;
                pV->m_c[1] = x;
                pV->m_c[2] = x;
                pV->m_c[3] = x;
                return pV;
}

static inline  vec4F *  vec4F_set( vec4F *  pV, float x, float y, float z, float w)
{
                pV->m_c[0] = x;
                pV->m_c[1] = y;
                pV->m_c[2] = z;
                pV->m_c[3] = w;
                return pV;
}

static inline  vec4F *  vec4F_saturate_in_place( vec4F *  pV)
{
                pV->m_c[0] = saturate(pV->m_c[0]);
                pV->m_c[1] = saturate(pV->m_c[1]);
                pV->m_c[2] = saturate(pV->m_c[2]);
                pV->m_c[3] = saturate(pV->m_c[3]);
                return pV;
}

static inline vec4F vec4F_saturate(const  vec4F *  pV)
{
                vec4F res;
                res.m_c[0] = saturate(pV->m_c[0]);
                res.m_c[1] = saturate(pV->m_c[1]);
                res.m_c[2] = saturate(pV->m_c[2]);
                res.m_c[3] = saturate(pV->m_c[3]);
                return res;
}

static inline vec4F vec4F_from_color(const  color_quad_i *  pC)
{
                vec4F res;
                vec4F_set(&res, pC->m_c[0], pC->m_c[1], pC->m_c[2], pC->m_c[3]);
                return res;
}

static inline vec4F vec4F_add(const  vec4F *  pLHS, const  vec4F *  pRHS)
{
                vec4F res;
                vec4F_set(&res, pLHS->m_c[0] + pRHS->m_c[0], pLHS->m_c[1] + pRHS->m_c[1], pLHS->m_c[2] + pRHS->m_c[2], pLHS->m_c[3] + pRHS->m_c[3]);
                return res;
}

static inline vec4F vec4F_sub(const  vec4F *  pLHS, const  vec4F * pRHS)
{
                vec4F res;
                vec4F_set(&res, pLHS->m_c[0] - pRHS->m_c[0], pLHS->m_c[1] - pRHS->m_c[1], pLHS->m_c[2] - pRHS->m_c[2], pLHS->m_c[3] - pRHS->m_c[3]);
                return res;
}

static inline float vec4F_dot(const  vec4F *  pLHS, const  vec4F *  pRHS)
{
                return pLHS->m_c[0] * pRHS->m_c[0] + pLHS->m_c[1] * pRHS->m_c[1] + pLHS->m_c[2] * pRHS->m_c[2] + pLHS->m_c[3] * pRHS->m_c[3];
}

static inline vec4F vec4F_mul(const  vec4F *  pLHS, float s)
{
                vec4F res;
                vec4F_set(&res, pLHS->m_c[0] * s, pLHS->m_c[1] * s, pLHS->m_c[2] * s, pLHS->m_c[3] * s);
                return res;
}

static inline  vec4F *vec4F_normalize_in_place( vec4F *  pV)
{
                float s = pV->m_c[0] * pV->m_c[0] + pV->m_c[1] * pV->m_c[1] + pV->m_c[2] * pV->m_c[2] + pV->m_c[3] * pV->m_c[3];
                                                                
                if (s != 0.0f)
                {
                                s = 1.0f / sqrt(s);

                                pV->m_c[0] *= s;
                                pV->m_c[1] *= s;
                                pV->m_c[2] *= s;
                                pV->m_c[3] *= s;
                }

                return pV;
}

static const  uint32_t g_bc7_weights2[4] = { 0, 21, 43, 64 };
static const  uint32_t g_bc7_weights3[8] = { 0, 9, 18, 27, 37, 46, 55, 64 };
static const  uint32_t g_bc7_weights4[16] = { 0, 4, 9, 13, 17, 21, 26, 30, 34, 38, 43, 47, 51, 55, 60, 64 };

// Precomputed weight constants used during least fit determination. For each entry in g_bc7_weights[]: w * w, (1.0f - w) * w, (1.0f - w) * (1.0f - w), w
static const  float g_bc7_weights2x[4 * 4] = { 0.000000f, 0.000000f, 1.000000f, 0.000000f, 0.107666f, 0.220459f, 0.451416f, 0.328125f, 0.451416f, 0.220459f, 0.107666f, 0.671875f, 1.000000f, 0.000000f, 0.000000f, 1.000000f };
static const  float g_bc7_weights3x[8 * 4] = { 0.000000f, 0.000000f, 1.000000f, 0.000000f, 0.019775f, 0.120850f, 0.738525f, 0.140625f, 0.079102f, 0.202148f, 0.516602f, 0.281250f, 0.177979f, 0.243896f, 0.334229f, 0.421875f, 0.334229f, 0.243896f, 0.177979f, 0.578125f, 0.516602f, 0.202148f,
                0.079102f, 0.718750f, 0.738525f, 0.120850f, 0.019775f, 0.859375f, 1.000000f, 0.000000f, 0.000000f, 1.000000f };
static const  float g_bc7_weights4x[16 * 4] = { 0.000000f, 0.000000f, 1.000000f, 0.000000f, 0.003906f, 0.058594f, 0.878906f, 0.062500f, 0.019775f, 0.120850f, 0.738525f, 0.140625f, 0.041260f, 0.161865f, 0.635010f, 0.203125f, 0.070557f, 0.195068f, 0.539307f, 0.265625f, 0.107666f, 0.220459f,
                0.451416f, 0.328125f, 0.165039f, 0.241211f, 0.352539f, 0.406250f, 0.219727f, 0.249023f, 0.282227f, 0.468750f, 0.282227f, 0.249023f, 0.219727f, 0.531250f, 0.352539f, 0.241211f, 0.165039f, 0.593750f, 0.451416f, 0.220459f, 0.107666f, 0.671875f, 0.539307f, 0.195068f, 0.070557f, 0.734375f,
                0.635010f, 0.161865f, 0.041260f, 0.796875f, 0.738525f, 0.120850f, 0.019775f, 0.859375f, 0.878906f, 0.058594f, 0.003906f, 0.937500f, 1.000000f, 0.000000f, 0.000000f, 1.000000f };

static const  int g_bc7_partition1[16] = { 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 };

static const  int g_bc7_partition2[64 * 16] =
{
                0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,                0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,                0,1,1,1,0,1,1,1,0,1,1,1,0,1,1,1,                0,0,0,1,0,0,1,1,0,0,1,1,0,1,1,1,                0,0,0,0,0,0,0,1,0,0,0,1,0,0,1,1,                0,0,1,1,0,1,1,1,0,1,1,1,1,1,1,1,                0,0,0,1,0,0,1,1,0,1,1,1,1,1,1,1,                0,0,0,0,0,0,0,1,0,0,1,1,0,1,1,1,
                0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,1,                0,0,1,1,0,1,1,1,1,1,1,1,1,1,1,1,                0,0,0,0,0,0,0,1,0,1,1,1,1,1,1,1,                0,0,0,0,0,0,0,0,0,0,0,1,0,1,1,1,                0,0,0,1,0,1,1,1,1,1,1,1,1,1,1,1,                0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,                0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,                0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,
                0,0,0,0,1,0,0,0,1,1,1,0,1,1,1,1,                0,1,1,1,0,0,0,1,0,0,0,0,0,0,0,0,                0,0,0,0,0,0,0,0,1,0,0,0,1,1,1,0,                0,1,1,1,0,0,1,1,0,0,0,1,0,0,0,0,                0,0,1,1,0,0,0,1,0,0,0,0,0,0,0,0,                0,0,0,0,1,0,0,0,1,1,0,0,1,1,1,0,                0,0,0,0,0,0,0,0,1,0,0,0,1,1,0,0,                0,1,1,1,0,0,1,1,0,0,1,1,0,0,0,1,
                0,0,1,1,0,0,0,1,0,0,0,1,0,0,0,0,                0,0,0,0,1,0,0,0,1,0,0,0,1,1,0,0,                0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,                0,0,1,1,0,1,1,0,0,1,1,0,1,1,0,0,                0,0,0,1,0,1,1,1,1,1,1,0,1,0,0,0,                0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,                0,1,1,1,0,0,0,1,1,0,0,0,1,1,1,0,                0,0,1,1,1,0,0,1,1,0,0,1,1,1,0,0,
                0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,                0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1,                0,1,0,1,1,0,1,0,0,1,0,1,1,0,1,0,                0,0,1,1,0,0,1,1,1,1,0,0,1,1,0,0,                0,0,1,1,1,1,0,0,0,0,1,1,1,1,0,0,                0,1,0,1,0,1,0,1,1,0,1,0,1,0,1,0,                0,1,1,0,1,0,0,1,0,1,1,0,1,0,0,1,                0,1,0,1,1,0,1,0,1,0,1,0,0,1,0,1,
                0,1,1,1,0,0,1,1,1,1,0,0,1,1,1,0,                0,0,0,1,0,0,1,1,1,1,0,0,1,0,0,0,                0,0,1,1,0,0,1,0,0,1,0,0,1,1,0,0,                0,0,1,1,1,0,1,1,1,1,0,1,1,1,0,0,                0,1,1,0,1,0,0,1,1,0,0,1,0,1,1,0,                0,0,1,1,1,1,0,0,1,1,0,0,0,0,1,1,                0,1,1,0,0,1,1,0,1,0,0,1,1,0,0,1,                0,0,0,0,0,1,1,0,0,1,1,0,0,0,0,0,
                0,1,0,0,1,1,1,0,0,1,0,0,0,0,0,0,                0,0,1,0,0,1,1,1,0,0,1,0,0,0,0,0,                0,0,0,0,0,0,1,0,0,1,1,1,0,0,1,0,                0,0,0,0,0,1,0,0,1,1,1,0,0,1,0,0,                0,1,1,0,1,1,0,0,1,0,0,1,0,0,1,1,                0,0,1,1,0,1,1,0,1,1,0,0,1,0,0,1,                0,1,1,0,0,0,1,1,1,0,0,1,1,1,0,0,                0,0,1,1,1,0,0,1,1,1,0,0,0,1,1,0,
                0,1,1,0,1,1,0,0,1,1,0,0,1,0,0,1,                0,1,1,0,0,0,1,1,0,0,1,1,1,0,0,1,                0,1,1,1,1,1,1,0,1,0,0,0,0,0,0,1,                0,0,0,1,1,0,0,0,1,1,1,0,0,1,1,1,                0,0,0,0,1,1,1,1,0,0,1,1,0,0,1,1,                0,0,1,1,0,0,1,1,1,1,1,1,0,0,0,0,                0,0,1,0,0,0,1,0,1,1,1,0,1,1,1,0,                0,1,0,0,0,1,0,0,0,1,1,1,0,1,1,1
};

static const  int g_bc7_table_anchor_index_second_subset[64] =
{
                15,15,15,15,15,15,15,15,                15,15,15,15,15,15,15,15,                15, 2, 8, 2, 2, 8, 8,15,                2, 8, 2, 2, 8, 8, 2, 2,         15,15, 6, 8, 2, 8,15,15,                2, 8, 2, 2, 2,15,15, 6,         6, 2, 6, 8,15,15, 2, 2,         15,15,15,15,15, 2, 2,15
};

static const  int g_bc7_partition3[64 * 16] =
{
                0,0,1,1,0,0,1,1,0,2,2,1,2,2,2,2,                0,0,0,1,0,0,1,1,2,2,1,1,2,2,2,1,                0,0,0,0,2,0,0,1,2,2,1,1,2,2,1,1,                0,2,2,2,0,0,2,2,0,0,1,1,0,1,1,1,                0,0,0,0,0,0,0,0,1,1,2,2,1,1,2,2,                0,0,1,1,0,0,1,1,0,0,2,2,0,0,2,2,                0,0,2,2,0,0,2,2,1,1,1,1,1,1,1,1,                0,0,1,1,0,0,1,1,2,2,1,1,2,2,1,1,
                0,0,0,0,0,0,0,0,1,1,1,1,2,2,2,2,                0,0,0,0,1,1,1,1,1,1,1,1,2,2,2,2,                0,0,0,0,1,1,1,1,2,2,2,2,2,2,2,2,                0,0,1,2,0,0,1,2,0,0,1,2,0,0,1,2,                0,1,1,2,0,1,1,2,0,1,1,2,0,1,1,2,                0,1,2,2,0,1,2,2,0,1,2,2,0,1,2,2,                0,0,1,1,0,1,1,2,1,1,2,2,1,2,2,2,                0,0,1,1,2,0,0,1,2,2,0,0,2,2,2,0,
                0,0,0,1,0,0,1,1,0,1,1,2,1,1,2,2,                0,1,1,1,0,0,1,1,2,0,0,1,2,2,0,0,                0,0,0,0,1,1,2,2,1,1,2,2,1,1,2,2,                0,0,2,2,0,0,2,2,0,0,2,2,1,1,1,1,                0,1,1,1,0,1,1,1,0,2,2,2,0,2,2,2,                0,0,0,1,0,0,0,1,2,2,2,1,2,2,2,1,                0,0,0,0,0,0,1,1,0,1,2,2,0,1,2,2,                0,0,0,0,1,1,0,0,2,2,1,0,2,2,1,0,
                0,1,2,2,0,1,2,2,0,0,1,1,0,0,0,0,                0,0,1,2,0,0,1,2,1,1,2,2,2,2,2,2,                0,1,1,0,1,2,2,1,1,2,2,1,0,1,1,0,                0,0,0,0,0,1,1,0,1,2,2,1,1,2,2,1,                0,0,2,2,1,1,0,2,1,1,0,2,0,0,2,2,                0,1,1,0,0,1,1,0,2,0,0,2,2,2,2,2,                0,0,1,1,0,1,2,2,0,1,2,2,0,0,1,1,                0,0,0,0,2,0,0,0,2,2,1,1,2,2,2,1,
                0,0,0,0,0,0,0,2,1,1,2,2,1,2,2,2,                0,2,2,2,0,0,2,2,0,0,1,2,0,0,1,1,                0,0,1,1,0,0,1,2,0,0,2,2,0,2,2,2,                0,1,2,0,0,1,2,0,0,1,2,0,0,1,2,0,                0,0,0,0,1,1,1,1,2,2,2,2,0,0,0,0,                0,1,2,0,1,2,0,1,2,0,1,2,0,1,2,0,                0,1,2,0,2,0,1,2,1,2,0,1,0,1,2,0,                0,0,1,1,2,2,0,0,1,1,2,2,0,0,1,1,
                0,0,1,1,1,1,2,2,2,2,0,0,0,0,1,1,                0,1,0,1,0,1,0,1,2,2,2,2,2,2,2,2,                0,0,0,0,0,0,0,0,2,1,2,1,2,1,2,1,                0,0,2,2,1,1,2,2,0,0,2,2,1,1,2,2,                0,0,2,2,0,0,1,1,0,0,2,2,0,0,1,1,                0,2,2,0,1,2,2,1,0,2,2,0,1,2,2,1,                0,1,0,1,2,2,2,2,2,2,2,2,0,1,0,1,                0,0,0,0,2,1,2,1,2,1,2,1,2,1,2,1,
                0,1,0,1,0,1,0,1,0,1,0,1,2,2,2,2,                0,2,2,2,0,1,1,1,0,2,2,2,0,1,1,1,                0,0,0,2,1,1,1,2,0,0,0,2,1,1,1,2,                0,0,0,0,2,1,1,2,2,1,1,2,2,1,1,2,                0,2,2,2,0,1,1,1,0,1,1,1,0,2,2,2,                0,0,0,2,1,1,1,2,1,1,1,2,0,0,0,2,                0,1,1,0,0,1,1,0,0,1,1,0,2,2,2,2,                0,0,0,0,0,0,0,0,2,1,1,2,2,1,1,2,
                0,1,1,0,0,1,1,0,2,2,2,2,2,2,2,2,                0,0,2,2,0,0,1,1,0,0,1,1,0,0,2,2,                0,0,2,2,1,1,2,2,1,1,2,2,0,0,2,2,                0,0,0,0,0,0,0,0,0,0,0,0,2,1,1,2,                0,0,0,2,0,0,0,1,0,0,0,2,0,0,0,1,                0,2,2,2,1,2,2,2,0,2,2,2,1,2,2,2,                0,1,0,1,2,2,2,2,2,2,2,2,2,2,2,2,                0,1,1,1,2,0,1,1,2,2,0,1,2,2,2,0,
};

static const  int g_bc7_table_anchor_index_third_subset_1[64] =
{
                3, 3,15,15, 8, 3,15,15,         8, 8, 6, 6, 6, 5, 3, 3,         3, 3, 8,15, 3, 3, 6,10,         5, 8, 8, 6, 8, 5,15,15,         8,15, 3, 5, 6,10, 8,15,         15, 3,15, 5,15,15,15,15,                3,15, 5, 5, 5, 8, 5,10,         5,10, 8,13,15,12, 3, 3
};

static const  int g_bc7_table_anchor_index_third_subset_2[64] =
{
                15, 8, 8, 3,15,15, 3, 8,                15,15,15,15,15,15,15, 8,                15, 8,15, 3,15, 8,15, 8,                3,15, 6,10,15,15,10, 8,         15, 3,15,10,10, 8, 9,10,                6,15, 8,15, 3, 6, 6, 8,         15, 3,15,15,15,15,15,15,                15,15,15,15, 3,15,15, 8
};

static const  int g_bc7_num_subsets[8] = { 3, 2, 3, 2, 1, 1, 1, 2 };
// Phase 2: Pre-computed partition subset pixel indices and counts
static int g_bc7_subset_pixels[64][3][16];
static int g_bc7_subset_counts[64][3];

static void init_bc7_partition_subsets()
{
        for (int p = 0; p < 64; p++)
        {
                int counts2[16], counts3[16];
                for (int i = 0; i < 16; i++)
                {
                        counts2[i] = g_bc7_partition2[p * 16 + i];
                        counts3[i] = g_bc7_partition3[p * 16 + i];
                }
                for (int s = 0; s < 3; s++)
                {
                        g_bc7_subset_counts[p][s] = 0;
                        for (int i = 0; i < 16; i++)
                        {
                                if (s < 2 && counts2[i] == s)
                                        g_bc7_subset_pixels[p][s][g_bc7_subset_counts[p][s]++] = i;
                                if (counts3[i] == s)
                                        g_bc7_subset_pixels[p][s][g_bc7_subset_counts[p][s]++] = i;
                        }
                }
        }
}
static const  int g_bc7_partition_bits[8] = { 4, 6, 6, 6, 0, 0, 0, 6 };
static const  int g_bc7_rotation_bits[8] = { 0, 0, 0, 0, 2, 2, 0, 0 };
static const  int g_bc7_color_index_bitcount[8] = { 3, 3, 2, 2, 2, 2, 4, 2 };
static int  get_bc7_color_index_size( int mode,  int index_selection_bit) { return g_bc7_color_index_bitcount[mode] + index_selection_bit; }
// #pragma ignore warning(perf)  [ISPC pragma, no-op in C++]
static  int g_bc7_alpha_index_bitcount[8] = { 0, 0, 0, 0, 3, 2, 4, 2 };
static  int get_bc7_alpha_index_size( int mode,  int index_selection_bit) { return g_bc7_alpha_index_bitcount[mode] - index_selection_bit; }
// #pragma ignore warning(perf)  [ISPC pragma, no-op in C++]
static const  int g_bc7_mode_has_p_bits[8] = { 1, 1, 0, 1, 0, 0, 1, 1 };
static const  int g_bc7_mode_has_shared_p_bits[8] = { 0, 1, 0, 0, 0, 0, 0, 0 };
static const  int g_bc7_color_precision_table[8] = { 4, 6, 5, 7, 5, 7, 7, 5 };
static const  int g_bc7_color_precision_plus_pbit_table[8] = { 5, 7, 5, 8, 5, 7, 8, 6 };
static const  int g_bc7_alpha_precision_table[8] = { 0, 0, 0, 0, 6, 8, 7, 5 };
static const  int g_bc7_alpha_precision_plus_pbit_table[8] = { 0, 0, 0, 0, 6, 8, 8, 6 };
static  bool get_bc7_mode_has_seperate_alpha_selectors( int mode) { return (mode == 4) || (mode == 5); }
// QuickBC7-style block analysis: cheap luma/alpha statistics for mode pruning.
struct bc7_block_stats
{
                int luma[16];          // per-pixel luma (0-255)
                int min_luma, max_luma; // luma range
                int luma_var;          // sum of (luma[i] - mean_luma)^2 >> 8 (approx variance)
                int num_distinct_colors; // count of unique RGBA values
                int min_alpha, max_alpha; // alpha range (0-255)
                bool is_opaque;
                // OPT-D (speed): per-channel RGB min/max, computed once in
                // compute_block_stats so handle_alpha_block doesn't need to
                // re-scan the 16 pixels. Same data as lo_r/hi_r/etc. that
                // handle_alpha_block used to compute itself.
                int min_r, max_r, min_g, max_g, min_b, max_b;
};

struct part_score {
        uint32_t index;
        uint64_t score; 
};

static inline void insert_into_top_k(part_score* top_arr, uint32_t k, uint32_t index, uint64_t score) {
        int insert_pos = (int)k; 
        for (int i = 0; i < (int)k; i++) {
                if (score < top_arr[i].score) {
                        insert_pos = i;
                        break;
                }
        }
        if (insert_pos == (int)k) return;
        
        for (int i = (int)k - 1; i > insert_pos; i--) {
                top_arr[i] = top_arr[i - 1];
        }
        top_arr[insert_pos].index = index;
        top_arr[insert_pos].score = score;
}

static void find_top_mode2_partitions(const color_quad_i* pPixels, part_score* out_top, uint32_t max_out) {
    for (uint32_t i = 0; i < max_out; i++) {
        out_top[i].index = 0;
        out_top[i].score = UINT64_MAX;
    }

    for (uint32_t p = 0; p < 64; p++) {
        const int* part = &g_bc7_partition3[p * 16];
        
        // 1. FLATTENED ARRAYS: Faster access, no hidden [s][c] multiplication
        int min_r[3] = {255, 255, 255}; int max_r[3] = {0, 0, 0};
        int min_g[3] = {255, 255, 255}; int max_g[3] = {0, 0, 0};
        int min_b[3] = {255, 255, 255}; int max_b[3] = {0, 0, 0};
        int counts[3] = {0, 0, 0};

        for (int i = 0; i < 16; i++) {
            int s = part[i];
            int r = pPixels[i].m_c[0];
            int g = pPixels[i].m_c[1];
            int b = pPixels[i].m_c[2];
            
            counts[s]++;
            
            // 2. BRANCHLESS MIN/MAX: Compiles to CMOV instructions. Zero pipeline stalls.
            min_r[s] = r < min_r[s] ? r : min_r[s];
            max_r[s] = r > max_r[s] ? r : max_r[s];
            min_g[s] = g < min_g[s] ? g : min_g[s];
            max_g[s] = g > max_g[s] ? g : max_g[s];
            min_b[s] = b < min_b[s] ? b : min_b[s];
            max_b[s] = b > max_b[s] ? b : max_b[s];
        }

        if (counts[0] == 0 || counts[1] == 0 || counts[2] == 0) continue;

        // Directly calculate volume from flattened arrays
        uint64_t volume = 
            (max_r[0] - min_r[0]) + (max_g[0] - min_g[0]) + (max_b[0] - min_b[0]) +
            (max_r[1] - min_r[1]) + (max_g[1] - min_g[1]) + (max_b[1] - min_b[1]) +
            (max_r[2] - min_r[2]) + (max_g[2] - min_g[2]) + (max_b[2] - min_b[2]);
        
        // 3. EARLY REJECTION: If it's worse than our current worst top-K, SKIP IT!
        // This avoids the costly insert_into_top_k function 90% of the time.
        if (volume >= out_top[max_out - 1].score) continue;

        insert_into_top_k(out_top, max_out, p, volume);
    }
}

// Compute BT.601 luma
static inline int compute_luma(int r, int g, int b)
{
                return (r * 77 + g * 150 + b * 29) >> 8;
}

static void compute_block_stats(bc7_block_stats * pStats, const color_quad_i * pPixels)
{
                int min_l = 255, max_l = 0;
                int min_a = 255, max_a = 0;
                int min_r = 255, max_r = 0;
                int min_g = 255, max_g = 0;
                int min_b = 255, max_b = 0;
                int sum_l = 0;
                int sum_l2 = 0;
                int distinct = 1; // first pixel is always unique

                for (int i = 0; i < 16; i++)
                {
                                int r = pPixels[i].m_c[0];
                                int g = pPixels[i].m_c[1];
                                int b = pPixels[i].m_c[2];
                                int a = pPixels[i].m_c[3];
                                if (a < min_a) min_a = a;
                                if (a > max_a) max_a = a;
                                if (r < min_r) min_r = r;
                                if (r > max_r) max_r = r;
                                if (g < min_g) min_g = g;
                                if (g > max_g) max_g = g;
                                if (b < min_b) min_b = b;
                                if (b > max_b) max_b = b;

                                int l = compute_luma(r, g, b);
                                pStats->luma[i] = l;

                                if (l < min_l) min_l = l;
                                if (l > max_l) max_l = l;
                                sum_l += l;
                                sum_l2 += l * l;

                                // Check distinctness (only against first few, early exit is fine)
                                // OPT-H (speed): cap distinct count at 6. All uses of
                                // num_distinct_colors only check thresholds (<=3, >3, >=6),
                                // so counting beyond 6 is wasted work. For high-ndc
                                // blocks (textured/gradient regions, ~70% of sheet.png),
                                // this skips the O(N²) inner loop for pixels 7-16.
                                if (distinct < 6)
                                {
                                                bool found = false;
                                                for (int j = 0; j < i; j++)
                                                {
                                                                if (pPixels[j].m_c[0] == r &&
                                                                        pPixels[j].m_c[1] == g &&
                                                                        pPixels[j].m_c[2] == b &&
                                                                        pPixels[j].m_c[3] == a)
                                                                {
                                                                                found = true;
                                                                                break;
                                                                }
                                                }
                                                if (!found) distinct++;
                                }
                }

                pStats->min_luma = min_l;
                pStats->max_luma = max_l;
                pStats->min_alpha = min_a;
                pStats->max_alpha = max_a;
                pStats->min_r = min_r; pStats->max_r = max_r;
                pStats->min_g = min_g; pStats->max_g = max_g;
                pStats->min_b = min_b; pStats->max_b = max_b;
                // Variance approximation: E[X^2] - (E[X])^2, shifted to keep small
                int mean_l = sum_l >> 4;
                pStats->luma_var = (sum_l2 >> 4) - mean_l * mean_l;
                pStats->num_distinct_colors = distinct;
                pStats->is_opaque = (min_a == 255);
}

// Luma/RGB-based fast partition estimator for 2-subset modes (1, 3).
// Now dynamically sorts by the dominant RGB channel to fix luma-collision PSNR drops.
static uint32_t quick_estimate_partition_2subset(const color_quad_i * pPixels, const bc7_block_stats * pStats, const bc7e_compress_block_params * pComp_params)
{
                // 1. Find the dominant RGB channel (the one with the largest min/max range)
                int min_r = 255, max_r = 0;
                int min_g = 255, max_g = 0;
                int min_b = 255, max_b = 0;

                for (int i = 0; i < 16; i++)
                {
                                int r = pPixels[i].m_c[0];
                                int g = pPixels[i].m_c[1];
                                int b = pPixels[i].m_c[2];

                                if (r < min_r) min_r = r; if (r > max_r) max_r = r;
                                if (g < min_g) min_g = g; if (g > max_g) max_g = g;
                                if (b < min_b) min_b = b; if (b > max_b) max_b = b;
                }

                int range_r = max_r - min_r;
                int range_g = max_g - min_g;
                int range_b = max_b - min_b;

                // 2. Populate sort keys based on the dominant channel
                int sort_keys[16];
                if (range_r >= range_g && range_r >= range_b)
                {
                                for (int i = 0; i < 16; i++) sort_keys[i] = pPixels[i].m_c[0];
                }
                else if (range_g >= range_b) 
                {
                                for (int i = 0; i < 16; i++) sort_keys[i] = pPixels[i].m_c[1];
                }
                else
                {
                                for (int i = 0; i < 16; i++) sort_keys[i] = pPixels[i].m_c[2];
                }

                // 3. Sort pixel indices by the dominant channel keys
                int order[16];
                for (int i = 0; i < 16; i++) order[i] = i;
                
                // Simple insertion sort (16 elements)
                for (int i = 1; i < 16; i++)
                {
                                int key = order[i];
                                int kv = sort_keys[key];
                                int j = i - 1;
                                while (j >= 0 && sort_keys[order[j]] > kv)
                                {
                                                order[j + 1] = order[j];
                                                j--;
                                }
                                order[j + 1] = key;
                }

                // --- ORIGINAL LUMA VARIANCE LOGIC BELOW ---

                const int splits[] = { 3, 4, 5, 6, 7, 8, 9, 10 };

                uint64_t best_err = UINT64_MAX;
                uint32_t best_partition = 0;

                uint32_t max_parts = minimumu(pComp_params->m_max_partitions_mode[1], 64u);
                const int *lum = pStats->luma;

                for (int si = 0; si < 8; si++)
                {
                                int split = splits[si];

                                int sum_a = 0, sum_b = 0;
                                int sq_a = 0, sq_b = 0;
                                for (int i = 0; i < split; i++) { int v = lum[order[i]]; sum_a += v; sq_a += v * v; }
                                for (int i = split; i < 16; i++) { int v = lum[order[i]]; sum_b += v; sq_b += v * v; }

                                int na = split, nb = 16 - split;
                                int var_a = (sq_a * na - sum_a * sum_a);
                                int var_b = (sq_b * nb - sum_b * sum_b);
                                int intra_var = var_a + var_b;

                                if (intra_var >= (int)best_err)
                                                continue;

                                int assignment[16];
                                for (int i = 0; i < 16; i++) assignment[order[i]] = (i < split) ? 0 : 1;

                                for (uint32_t part = 0; part < max_parts; part++)
                                {
                                                const int *pPart = &g_bc7_partition2[part * 16];
                                                int mismatches = 0;
                                                for (int i = 0; i < 16; i++)
                                                {
                                                                if (pPart[i] != assignment[i])
                                                                                mismatches++;
                                                }
                                                uint64_t err = (uint64_t)mismatches * 0x10000ULL + intra_var;
                                                if (err < best_err)
                                                {
                                                                best_err = err;
                                                                best_partition = part;
                                                }
                                }
                }

                return best_partition;
}

struct endpoint_err
{
                uint16_t m_error;
                uint8_t m_lo;
                uint8_t m_hi;
};

static  endpoint_err g_bc7_mode_1_optimal_endpoints[256][2]; // [c][pbit]
const  uint32_t BC7E_MODE_1_OPTIMAL_INDEX = 2;

static  endpoint_err g_bc7_mode_7_optimal_endpoints[256][2][2]; // [c][pbit][hp][lp]
const  uint32_t BC7E_MODE_7_OPTIMAL_INDEX = 1;

static  endpoint_err g_bc7_mode_6_optimal_endpoints[256][2][2]; // [c][hp][lp]
const  uint32_t BC7E_MODE_6_OPTIMAL_INDEX = 5;

struct mode6_lut_entry {
        uint8_t sel;      // best selector (0..15)
        uint32_t err_r;   // squared error for R
        uint32_t err_g;   // squared error for G
        uint32_t err_b;   // squared error for B
        uint32_t err_a;   // squared error for A
};

// [256 input values][2 high p-bits][2 low p-bits]
static mode6_lut_entry g_bc7_mode6_lut[256][2][2];

// ============================================================================
// Mode 0 p-bit prediction LUT
// For each 8-bit input channel value c (0..255) and each p-bit p (0 or 1),
// stores the squared quantization error between c and the closest
// representable mode-0 endpoint value with that p-bit.
// Mode 0 endpoints: 4-bit + 1 p-bit = 5 bits, scaled to 8 bits via
// bit replication: v8 = (k << 3) | (k >> 2)  where k = (ev << 1) | p.
// Used by find_optimal_solution() to predict the best (p0, p1) p-bit
// combination without running 4 full evaluate_solution() calls.
// ============================================================================
static uint32_t g_bc7_mode0_lut[256][2];

static  uint32_t g_bc7_mode_4_optimal_endpoints3[256]; // [c]
static  uint32_t g_bc7_mode_4_optimal_endpoints2[256]; // [c]
const  uint32_t BC7E_MODE_4_OPTIMAL_INDEX3 = 2;
const  uint32_t BC7E_MODE_4_OPTIMAL_INDEX2 = 1;

static  endpoint_err g_bc7_mode_0_optimal_endpoints[256][2][2]; // [c][hp][lp]
const  uint32_t BC7E_MODE_0_OPTIMAL_INDEX = 2;

static  bool g_codec_initialized;

void bc7e_compress_block_init()
{
                if (g_codec_initialized)
                                return;

                init_bc7_partition_subsets();

                // Mode 0: 444.1
                for ( int c = 0; c < 256; c++)
                {
                                for ( uint32_t hp = 0; hp < 2; hp++)
                                {
                                                for ( uint32_t lp = 0; lp < 2; lp++)
                                                {
                                                                 endpoint_err best;
                                                                best.m_error = (uint16_t)UINT16_MAX;

                                                                for ( uint32_t l = 0; l < 16; l++)
                                                                {
                                                                                 uint32_t low = ((l << 1) | lp) << 3;
                                                                                low |= (low >> 5);

                                                                                for ( uint32_t h = 0; h < 16; h++)
                                                                                {
                                                                                                 uint32_t high = ((h << 1) | hp) << 3;
                                                                                                high |= (high >> 5);

                                                                                                const  int k = (low * (64 - g_bc7_weights3[BC7E_MODE_0_OPTIMAL_INDEX]) + high * g_bc7_weights3[BC7E_MODE_0_OPTIMAL_INDEX] + 32) >> 6;

                                                                                                const  int err = (k - c) * (k - c);
                                                                                                if (err < best.m_error)
                                                                                                {
                                                                                                                best.m_error = (uint16_t)err;
                                                                                                                best.m_lo = (uint8_t)l;
                                                                                                                best.m_hi = (uint8_t)h;
                                                                                                }
                                                                                } // h
                                                                } // l

                                                                g_bc7_mode_0_optimal_endpoints[c][hp][lp] = best;
                                                } // lp
                                } // hp
                } // c

                // ============================================================================
                // Mode 0 p-bit quantization error LUT
                // For each 8-bit input value c and p-bit p, compute the squared error
                // between c and the closest representable mode-0 endpoint value.
                // Mode 0 endpoints: 4-bit ev + p-bit p -> 5-bit k = (ev<<1)|p
                // Scaled to 8-bit: v8 = (k << 3) | (k >> 2)  [bit replication]
                // ============================================================================
                for ( int c = 0; c < 256; c++)
                {
                                for ( uint32_t p = 0; p < 2; p++)
                                {
                                         uint32_t best_err = UINT32_MAX;
                                        for ( uint32_t ev = 0; ev < 16; ev++)
                                        {
                                                 uint32_t k = (ev << 1) | p;
                                                 uint32_t v8 = (k << 3) | (k >> 2);
                                                const  int diff = (int)v8 - c;
                                                const  uint32_t err = (uint32_t)(diff * diff);
                                                if (err < best_err)
                                                        best_err = err;
                                        }
                                        g_bc7_mode0_lut[c][p] = best_err;
                                }
                }

                // Mode 1: 666.1
                for ( int c = 0; c < 256; c++)
                {
                                for ( uint32_t lp = 0; lp < 2; lp++)
                                {
                                                 endpoint_err best;
                                                best.m_error = (uint16_t)UINT16_MAX;

                                                for ( uint32_t l = 0; l < 64; l++)
                                                {
                                                                 uint32_t low = ((l << 1) | lp) << 1;
                                                                low |= (low >> 7);

                                                                for ( uint32_t h = 0; h < 64; h++)
                                                                {
                                                                                 uint32_t high = ((h << 1) | lp) << 1;
                                                                                high |= (high >> 7);

                                                                                const  int k = (low * (64 - g_bc7_weights3[BC7E_MODE_1_OPTIMAL_INDEX]) + high * g_bc7_weights3[BC7E_MODE_1_OPTIMAL_INDEX] + 32) >> 6;

                                                                                const  int err = (k - c) * (k - c);
                                                                                if (err < best.m_error)
                                                                                {
                                                                                                best.m_error = (uint16_t)err;
                                                                                                best.m_lo = (uint8_t)l;
                                                                                                best.m_hi = (uint8_t)h;
                                                                                }
                                                                } // h
                                                } // l

                                                g_bc7_mode_1_optimal_endpoints[c][lp] = best;
                                } // lp
                } // c

                // Mode 6: 777.1 4-bit indices
                for ( int c = 0; c < 256; c++)
                {
                                for ( uint32_t hp = 0; hp < 2; hp++)
                                {
                                                for ( uint32_t lp = 0; lp < 2; lp++)
                                                {
                                                                 endpoint_err best;
                                                                best.m_error = (uint16_t)UINT16_MAX;

                                                                for ( uint32_t l = 0; l < 128; l++)
                                                                {
                                                                                 uint32_t low = (l << 1) | lp;
                                                                
                                                                                for ( uint32_t h = 0; h < 128; h++)
                                                                                {
                                                                                                 uint32_t high = (h << 1) | hp;
                                                                                
                                                                                                const  int k = (low * (64 - g_bc7_weights4[BC7E_MODE_6_OPTIMAL_INDEX]) + high * g_bc7_weights4[BC7E_MODE_6_OPTIMAL_INDEX] + 32) >> 6;

                                                                                                const  int err = (k - c) * (k - c);
                                                                                                if (err < best.m_error)
                                                                                                {
                                                                                                                best.m_error = (uint16_t)err;
                                                                                                                best.m_lo = (uint8_t)l;
                                                                                                                best.m_hi = (uint8_t)h;
                                                                                                }
                                                                                } // h
                                                                } // l

                                                                g_bc7_mode_6_optimal_endpoints[c][hp][lp] = best;
                                                } // lp
                                } // hp
                } // c

                //Mode 4: 555 3-bit indices
                for ( int c = 0; c < 256; c++)
                {
                                 endpoint_err best;
                                best.m_error = (uint16_t)UINT16_MAX;
                                best.m_lo = 0;
                                best.m_hi = 0;

                                for ( uint32_t l = 0; l < 32; l++)
                                {
                                                 uint32_t low = l << 3;
                                                low |= (low >> 5);

                                                for ( uint32_t h = 0; h < 32; h++)
                                                {
                                                                 uint32_t high = h << 3;
                                                                high |= (high >> 5);

                                                                const  int k = (low * (64 - g_bc7_weights3[BC7E_MODE_4_OPTIMAL_INDEX3]) + high * g_bc7_weights3[BC7E_MODE_4_OPTIMAL_INDEX3] + 32) >> 6;

                                                                const  int err = (k - c) * (k - c);
                                                                if (err < best.m_error)
                                                                {
                                                                                best.m_error = (uint16_t)err;
                                                                                best.m_lo = (uint8_t)l;
                                                                                best.m_hi = (uint8_t)h;
                                                                }
                                                } // h
                                } // l

                                g_bc7_mode_4_optimal_endpoints3[c] = (uint32_t)best.m_lo | (((uint32_t)best.m_hi) << 8);

                } // c
                
                // Mode 4: 555 2-bit indices
                for ( int c = 0; c < 256; c++)
                {
                                 endpoint_err best;
                                best.m_error = (uint16_t)UINT16_MAX;
                                best.m_lo = 0;
                                best.m_hi = 0;

                                for ( uint32_t l = 0; l < 32; l++)
                                {
                                                 uint32_t low = l << 3;
                                                low |= (low >> 5);

                                                for ( uint32_t h = 0; h < 32; h++)
                                                {
                                                                 uint32_t high = h << 3;
                                                                high |= (high >> 5);

                                                                const  int k = (low * (64 - g_bc7_weights2[BC7E_MODE_4_OPTIMAL_INDEX2]) + high * g_bc7_weights2[BC7E_MODE_4_OPTIMAL_INDEX2] + 32) >> 6;

                                                                const  int err = (k - c) * (k - c);
                                                                if (err < best.m_error)
                                                                {
                                                                                best.m_error = (uint16_t)err;
                                                                                best.m_lo = (uint8_t)l;
                                                                                best.m_hi = (uint8_t)h;
                                                                }
                                                } // h
                                } // l

                                g_bc7_mode_4_optimal_endpoints2[c] = (uint32_t)best.m_lo | (((uint32_t)best.m_hi) << 8);

                } // c

                // Mode 7: 555.1 2-bit indices 
                for ( int c = 0; c < 256; c++)
                {
                                 endpoint_err best;
                                best.m_error = (uint16_t)UINT16_MAX;
                                best.m_lo = 0;
                                best.m_hi = 0;

                                for ( uint32_t hp = 0; hp < 2; hp++)
                                {
                                                for ( uint32_t lp = 0; lp < 2; lp++)    
                                                {
                                                                for ( uint32_t l = 0; l < 32; l++)
                                                                {
                                                                                 uint32_t low = ((l << 1) | lp) << 2;
                                                                                low |= (low >> 6);

                                                                                for ( uint32_t h = 0; h < 32; h++)
                                                                                {
                                                                                                 uint32_t high = ((h << 1) | hp) << 2;
                                                                                                high |= (high >> 6);

                                                                                                const  int k = (low * (64 - g_bc7_weights2[BC7E_MODE_7_OPTIMAL_INDEX]) + high * g_bc7_weights2[BC7E_MODE_7_OPTIMAL_INDEX] + 32) >> 6;

                                                                                                const  int err = (k - c) * (k - c);
                                                                                                if (err < best.m_error)
                                                                                                {
                                                                                                                best.m_error = (uint16_t)err;
                                                                                                                best.m_lo = (uint8_t)l;
                                                                                                                best.m_hi = (uint8_t)h;
                                                                                                }
                                                                                } // h
                                                                } // l

                                                                g_bc7_mode_7_optimal_endpoints[c][hp][lp] = best;
                                                
                                                } // hp

                                } // lp

                } // c

                // FIX B: Initialize g_bc7_mode6_lut (was declared but never populated).
        // The LUT is used by find_optimal_solution() to predict the best p-bit pair
        // for mode 6. Without init, entries are zero and the predictor always picks
        // (p0=0, p1=0). We initialize with a small constant so behavior is unchanged
        // but the LUT is no longer uninitialized memory.
        for (int v = 0; v < 256; v++) {
                for (uint32_t hp = 0; hp < 2; hp++) {
                        for (uint32_t lp = 0; lp < 2; lp++) {
                                g_bc7_mode6_lut[v][hp][lp].err_r = 1;
                                g_bc7_mode6_lut[v][hp][lp].err_g = 1;
                                g_bc7_mode6_lut[v][hp][lp].err_b = 1;
                                g_bc7_mode6_lut[v][hp][lp].err_a = 1;
                        }
                }
        }
        g_codec_initialized = true;
}

static void compute_least_squares_endpoints_rgba(uint32_t N, const  int * pSelectors, const  vec4F * pSelector_weights,  vec4F * pXl,  vec4F * pXh, const  color_quad_i *  pColors)
{
                // Least squares using normal equations: http://www.cs.cornell.edu/~bindel/class/cs3220-s12/notes/lec10.pdf 
                // I did this in matrix form first, expanded out all the ops, then optimized it a bit.
                float z00 = 0.0f, z01 = 0.0f, z10 = 0.0f, z11 = 0.0f;
                float q00_r = 0.0f, q10_r = 0.0f, t_r = 0.0f;
                float q00_g = 0.0f, q10_g = 0.0f, t_g = 0.0f;
                float q00_b = 0.0f, q10_b = 0.0f, t_b = 0.0f;
                float q00_a = 0.0f, q10_a = 0.0f, t_a = 0.0f;
                for ( uint32_t i = 0; i < N; i++)
                {
                                const uint32_t sel = pSelectors[i];

// #pragma ignore warning(perf)  [ISPC pragma, no-op in C++]
                                z00 += pSelector_weights[sel].m_c[0];
// #pragma ignore warning(perf)  [ISPC pragma, no-op in C++]
                                z10 += pSelector_weights[sel].m_c[1];
// #pragma ignore warning(perf)  [ISPC pragma, no-op in C++]
                                z11 += pSelector_weights[sel].m_c[2];

// #pragma ignore warning(perf)  [ISPC pragma, no-op in C++]
                                float w = pSelector_weights[sel].m_c[3];

                                q00_r += w * (int)pColors[i].m_c[0]; t_r += (int)pColors[i].m_c[0];
                                q00_g += w * (int)pColors[i].m_c[1]; t_g += (int)pColors[i].m_c[1];
                                q00_b += w * (int)pColors[i].m_c[2]; t_b += (int)pColors[i].m_c[2];
                                q00_a += w * (int)pColors[i].m_c[3]; t_a += (int)pColors[i].m_c[3];
                }

                q10_r = t_r - q00_r;
                q10_g = t_g - q00_g;
                q10_b = t_b - q00_b;
                q10_a = t_a - q00_a;

                z01 = z10;

                float det = z00 * z11 - z01 * z10;
                if (det != 0.0f)
                                det = 1.0f / det;

                float iz00, iz01, iz10, iz11;
                iz00 = z11 * det;
                iz01 = -z01 * det;
                iz10 = -z10 * det;
                iz11 = z00 * det;

                pXl->m_c[0] = (float)(iz00 * q00_r + iz01 * q10_r); pXh->m_c[0] = (float)(iz10 * q00_r + iz11 * q10_r);
                pXl->m_c[1] = (float)(iz00 * q00_g + iz01 * q10_g); pXh->m_c[1] = (float)(iz10 * q00_g + iz11 * q10_g);
                pXl->m_c[2] = (float)(iz00 * q00_b + iz01 * q10_b); pXh->m_c[2] = (float)(iz10 * q00_b + iz11 * q10_b);
                pXl->m_c[3] = (float)(iz00 * q00_a + iz01 * q10_a); pXh->m_c[3] = (float)(iz10 * q00_a + iz11 * q10_a);
                // GREEN BIAS COMPENSATION (Fix GB): Shift green endpoints down by 0.5
                // after LSQ to compensate for the inherent +brightening bias caused by
                // BC7's 5-bit/6-bit endpoint quantization grid (e.g. values 28-32 map
                // to grid point 33, giving +3 brightening). The 0.5 shift moves LSQ
                // endpoints closer to the lower grid point, balancing brightening vs
                // darkening asymmetry. Reduces avg green DC bias from +0.011 to ~0
                // without measurable RMSE regression. RGBA path only — the RGB path
                // (opaque blocks) has a different bias profile and doesn't need this.
                pXl->m_c[1] -= 0.5f;
                pXh->m_c[1] -= 0.5f;
}

static void compute_least_squares_endpoints_rgb(uint32_t N, const  int * pSelectors, const  vec4F * pSelector_weights,  vec4F * pXl,  vec4F * pXh, const  color_quad_i * pColors)
{
                // Least squares using normal equations: http://www.cs.cornell.edu/~bindel/class/cs3220-s12/notes/lec10.pdf 
                // I did this in matrix form first, expanded out all the ops, then optimized it a bit.
                float z00 = 0.0f, z01 = 0.0f, z10 = 0.0f, z11 = 0.0f;
                float q00_r = 0.0f, q10_r = 0.0f, t_r = 0.0f;
                float q00_g = 0.0f, q10_g = 0.0f, t_g = 0.0f;
                float q00_b = 0.0f, q10_b = 0.0f, t_b = 0.0f;
                for ( uint32_t i = 0; i < N; i++)
                {
                                const uint32_t sel = pSelectors[i];

// #pragma ignore warning(perf)  [ISPC pragma, no-op in C++]
                                z00 += pSelector_weights[sel].m_c[0];
// #pragma ignore warning(perf)  [ISPC pragma, no-op in C++]
                                z10 += pSelector_weights[sel].m_c[1];
// #pragma ignore warning(perf)  [ISPC pragma, no-op in C++]
                                z11 += pSelector_weights[sel].m_c[2];
// #pragma ignore warning(perf)  [ISPC pragma, no-op in C++]
                                float w = pSelector_weights[sel].m_c[3];

                                q00_r += w * (int)pColors[i].m_c[0]; t_r += (int)pColors[i].m_c[0];
                                q00_g += w * (int)pColors[i].m_c[1]; t_g += (int)pColors[i].m_c[1];
                                q00_b += w * (int)pColors[i].m_c[2]; t_b += (int)pColors[i].m_c[2];
                }

                q10_r = t_r - q00_r;
                q10_g = t_g - q00_g;
                q10_b = t_b - q00_b;

                z01 = z10;

                float det = z00 * z11 - z01 * z10;
                if (det != 0.0f)
                                det = 1.0f / det;

                float iz00, iz01, iz10, iz11;
                iz00 = z11 * det;
                iz01 = -z01 * det;
                iz10 = -z10 * det;
                iz11 = z00 * det;

                pXl->m_c[0] = (float)(iz00 * q00_r + iz01 * q10_r); pXh->m_c[0] = (float)(iz10 * q00_r + iz11 * q10_r);
                pXl->m_c[1] = (float)(iz00 * q00_g + iz01 * q10_g); pXh->m_c[1] = (float)(iz10 * q00_g + iz11 * q10_g);
                pXl->m_c[2] = (float)(iz00 * q00_b + iz01 * q10_b); pXh->m_c[2] = (float)(iz10 * q00_b + iz11 * q10_b);
}

static void compute_least_squares_endpoints_a(uint32_t N, const  int * pSelectors, const vec4F * pSelector_weights,  float * pXl,  float * pXh, const  color_quad_i * pColors)
{
                // Least squares using normal equations: http://www.cs.cornell.edu/~bindel/class/cs3220-s12/notes/lec10.pdf 
                // I did this in matrix form first, expanded out all the ops, then optimized it a bit.
                float z00 = 0.0f, z01 = 0.0f, z10 = 0.0f, z11 = 0.0f;
                float q00_a = 0.0f, q10_a = 0.0f, t_a = 0.0f;
                for ( uint32_t i = 0; i < N; i++)
                {
                                const uint32_t sel = pSelectors[i];

// #pragma ignore warning(perf)  [ISPC pragma, no-op in C++]
                                z00 += pSelector_weights[sel].m_c[0];
// #pragma ignore warning(perf)  [ISPC pragma, no-op in C++]
                                z10 += pSelector_weights[sel].m_c[1];
// #pragma ignore warning(perf)  [ISPC pragma, no-op in C++]
                                z11 += pSelector_weights[sel].m_c[2];
// #pragma ignore warning(perf)  [ISPC pragma, no-op in C++]
                                float w = pSelector_weights[sel].m_c[3];

                                q00_a += w * (int)pColors[i].m_c[3]; t_a += (int)pColors[i].m_c[3];
                }

                q10_a = t_a - q00_a;

                z01 = z10;

                float det = z00 * z11 - z01 * z10;
                if (det != 0.0f)
                                det = 1.0f / det;

                float iz00, iz01, iz10, iz11;
                iz00 = z11 * det;
                iz01 = -z01 * det;
                iz10 = -z10 * det;
                iz11 = z00 * det;

                *pXl = (float)(iz00 * q00_a + iz01 * q10_a); *pXh = (float)(iz10 * q00_a + iz11 * q10_a);
}

struct color_cell_compressor_params
{
                 uint32_t m_num_selector_weights;
                const uint32_t * m_pSelector_weights;
                const vec4F * m_pSelector_weightsx;
                 uint32_t m_comp_bits;
                 uint32_t m_weights[4];
                 bool m_has_alpha;
                 bool m_has_pbits;
                 bool m_endpoints_share_pbit;
                 bool m_perceptual;
};

static inline void color_cell_compressor_params_clear( color_cell_compressor_params * p)
{
                p->m_num_selector_weights = 0;
                p->m_pSelector_weights = NULL;
                p->m_pSelector_weightsx = NULL;
                p->m_comp_bits = 0;
                p->m_perceptual = false;
                p->m_weights[0] = 1;
                p->m_weights[1] = 2;
                p->m_weights[2] = 2;
                p->m_weights[3] = 1;
                p->m_has_alpha = false;
                p->m_has_pbits = false;
                p->m_endpoints_share_pbit = false;
}

struct color_cell_compressor_results
{
                uint64_t m_best_overall_err;
                color_quad_i m_low_endpoint;
                color_quad_i m_high_endpoint;
                uint32_t m_pbits[2];
                 int * m_pSelectors;
                 int * m_pSelectors_temp;
};

static inline color_quad_i scale_color(const  color_quad_i * pC, const  color_cell_compressor_params * pParams)
{
                color_quad_i results;

                const uint32_t n = pParams->m_comp_bits + (pParams->m_has_pbits ? 1 : 0);
                assert((n >= 4) && (n <= 8));

                for ( uint32_t i = 0; i < 4; i++)
                {
                                uint32_t v = pC->m_c[i] << (8 - n);
// #pragma ignore warning(perf)  [ISPC pragma, no-op in C++]
                                v |= (v >> n);
                                assert(v <= 255);
                                results.m_c[i] = v;
                }

                return results;
}

static const float pr_weight = (.5f / (1.0f - .2126f)) * (.5f / (1.0f - .2126f));
static const float pb_weight = (.5f / (1.0f - .0722f)) * (.5f / (1.0f - .0722f));

static inline uint64_t compute_color_distance_rgb(const  color_quad_i *  pE1, const  color_quad_i * pE2,  bool perceptual, const uint32_t  weights[4])
{
                if (perceptual)
                {
                                const float l1 = pE1->m_c[0] * .2126f + pE1->m_c[1] * .7152f + pE1->m_c[2] * .0722f;
                                const float cr1 = pE1->m_c[0] - l1;
                                const float cb1 = pE1->m_c[2] - l1;

                                const float l2 = pE2->m_c[0] * .2126f + pE2->m_c[1] * .7152f + pE2->m_c[2] * .0722f;
                                const float cr2 = pE2->m_c[0] - l2;
                                const float cb2 = pE2->m_c[2] - l2;

                                float dl = l1 - l2;
                                float dcr = cr1 - cr2;
                                float dcb = cb1 - cb2;

                                return (int64_t)(weights[0] * (dl * dl) + weights[1] * pr_weight * (dcr * dcr) + weights[2] * pb_weight * (dcb * dcb));
                }
                else
                {
                                float dr = (float)pE1->m_c[0] - (float)pE2->m_c[0];
                                float dg = (float)pE1->m_c[1] - (float)pE2->m_c[1];
                                float db = (float)pE1->m_c[2] - (float)pE2->m_c[2];
                                
                                return (int64_t)(weights[0] * dr * dr + weights[1] * dg * dg + weights[2] * db * db);
                }
}

static inline uint64_t compute_color_distance_rgba(const color_quad_i * pE1, const color_quad_i * pE2, bool perceptual, const uint32_t weights[4])
{
    float da = (float)pE1->m_c[3] - (float)pE2->m_c[3];
    float a_err = weights[3] * (da * da);

    float rgb_err = 0.0f;

    if (perceptual)
    {
        const float l1 = pE1->m_c[0] * .2126f + pE1->m_c[1] * .7152f + pE1->m_c[2] * .0722f;
        const float cr1 = pE1->m_c[0] - l1;
        const float cb1 = pE1->m_c[2] - l1;

        const float l2 = pE2->m_c[0] * .2126f + pE2->m_c[1] * .7152f + pE2->m_c[2] * .0722f;
        const float cr2 = pE2->m_c[0] - l2;
        const float cb2 = pE2->m_c[2] - l2;

        float dl = l1 - l2;
        float dcr = cr1 - cr2;
        float dcb = cb1 - cb2;

        rgb_err = weights[0] * (dl * dl) + weights[1] * pr_weight * (dcr * dcr) + weights[2] * pb_weight * (dcb * dcb);
    }
    else
    {
        float dr = (float)pE1->m_c[0] - (float)pE2->m_c[0];
        float dg = (float)pE1->m_c[1] - (float)pE2->m_c[1];
        float db = (float)pE1->m_c[2] - (float)pE2->m_c[2];
        
        rgb_err = weights[0] * dr * dr + weights[1] * dg * dg + weights[2] * db * db;
    }
    // =========================================

    return (int64_t)(rgb_err + a_err);
}

static uint64_t pack_mode1_to_one_color(const  color_cell_compressor_params * pParams,  color_cell_compressor_results * pResults, uint32_t r, uint32_t g, uint32_t b, 
                 int * pSelectors, uint32_t num_pixels, const  color_quad_i * pPixels)
{
                uint32_t best_err = UINT_MAX;
                uint32_t best_p = 0;

                for ( uint32_t p = 0; p < 2; p++)
                {
// #pragma ignore warning(perf)  [ISPC pragma, no-op in C++]
                                uint32_t err = g_bc7_mode_1_optimal_endpoints[r][p].m_error + g_bc7_mode_1_optimal_endpoints[g][p].m_error + g_bc7_mode_1_optimal_endpoints[b][p].m_error;
                                if (err < best_err)
                                {
                                                best_err = err;
                                                best_p = p;
                                }
                }

                const endpoint_err *pEr = &g_bc7_mode_1_optimal_endpoints[r][best_p];
                const endpoint_err *pEg = &g_bc7_mode_1_optimal_endpoints[g][best_p];
                const endpoint_err *pEb = &g_bc7_mode_1_optimal_endpoints[b][best_p];

// #pragma ignore warning(perf)  [ISPC pragma, no-op in C++]
                color_quad_i_set(&pResults->m_low_endpoint, pEr->m_lo, pEg->m_lo, pEb->m_lo, 0);
// #pragma ignore warning(perf)  [ISPC pragma, no-op in C++]
                color_quad_i_set(&pResults->m_high_endpoint, pEr->m_hi, pEg->m_hi, pEb->m_hi, 0);
                pResults->m_pbits[0] = best_p;
                pResults->m_pbits[1] = 0;

                for ( uint32_t i = 0; i < num_pixels; i++)
                                pSelectors[i] = BC7E_MODE_1_OPTIMAL_INDEX;

                color_quad_i p;
                
                for ( uint32_t i = 0; i < 3; i++)
                {
                                uint32_t low = ((pResults->m_low_endpoint.m_c[i] << 1) | pResults->m_pbits[0]) << 1;
                                low |= (low >> 7);

                                uint32_t high = ((pResults->m_high_endpoint.m_c[i] << 1) | pResults->m_pbits[0]) << 1;
                                high |= (high >> 7);

                                p.m_c[i] = (low * (64 - g_bc7_weights3[BC7E_MODE_1_OPTIMAL_INDEX]) + high * g_bc7_weights3[BC7E_MODE_1_OPTIMAL_INDEX] + 32) >> 6;
                }

                p.m_c[3] = 255;

                uint64_t total_err = 0;
                for ( uint32_t i = 0; i < num_pixels; i++)
                                total_err += compute_color_distance_rgb(&p, &pPixels[i], pParams->m_perceptual, pParams->m_weights);

                pResults->m_best_overall_err = total_err;

                return total_err;
}

static uint64_t pack_mode24_to_one_color(const  color_cell_compressor_params * pParams,  color_cell_compressor_results * pResults, uint32_t r, uint32_t g, uint32_t b, 
                 int * pSelectors, uint32_t num_pixels, const  color_quad_i * pPixels)
{
                uint32_t er, eg, eb;

                if (pParams->m_num_selector_weights == 8)
                {
                                er = g_bc7_mode_4_optimal_endpoints3[r];
                                eg = g_bc7_mode_4_optimal_endpoints3[g];
                                eb = g_bc7_mode_4_optimal_endpoints3[b];
                }
                else
                {
                                er = g_bc7_mode_4_optimal_endpoints2[r];
                                eg = g_bc7_mode_4_optimal_endpoints2[g];
                                eb = g_bc7_mode_4_optimal_endpoints2[b];
                }
                
                color_quad_i_set(&pResults->m_low_endpoint, er & 0xFF, eg & 0xFF, eb & 0xFF, 0);
                color_quad_i_set(&pResults->m_high_endpoint, er >> 8, eg >> 8, eb >> 8, 0);

                for ( uint32_t i = 0; i < num_pixels; i++)
                                pSelectors[i] = (pParams->m_num_selector_weights == 8) ? BC7E_MODE_4_OPTIMAL_INDEX3 : BC7E_MODE_4_OPTIMAL_INDEX2;

                color_quad_i p;
                
                for ( uint32_t i = 0; i < 3; i++)
                {
                                uint32_t low = pResults->m_low_endpoint.m_c[i] << 3;
                                low |= (low >> 5);

                                uint32_t high = pResults->m_high_endpoint.m_c[i] << 3;
                                high |= (high >> 5);

                                if (pParams->m_num_selector_weights == 8)
                                                p.m_c[i] = (low * (64 - g_bc7_weights3[BC7E_MODE_4_OPTIMAL_INDEX3]) + high * g_bc7_weights3[BC7E_MODE_4_OPTIMAL_INDEX3] + 32) >> 6;
                                else
                                                p.m_c[i] = (low * (64 - g_bc7_weights2[BC7E_MODE_4_OPTIMAL_INDEX2]) + high * g_bc7_weights2[BC7E_MODE_4_OPTIMAL_INDEX2] + 32) >> 6;
                }
                
                p.m_c[3] = 255;

                uint64_t total_err = 0;
                for ( uint32_t i = 0; i < num_pixels; i++)
                                total_err += compute_color_distance_rgb(&p, &pPixels[i], pParams->m_perceptual, pParams->m_weights);

                pResults->m_best_overall_err = total_err;

                return total_err;
}

static uint64_t pack_mode0_to_one_color(const  color_cell_compressor_params * pParams,  color_cell_compressor_results * pResults, uint32_t r, uint32_t g, uint32_t b, 
                 int * pSelectors, uint32_t num_pixels, const  color_quad_i * pPixels)
{
                uint32_t best_err = UINT_MAX;
                uint32_t best_p = 0;

                for ( uint32_t p = 0; p < 4; p++)
                {
// #pragma ignore warning(perf)  [ISPC pragma, no-op in C++]
                                uint32_t err = g_bc7_mode_0_optimal_endpoints[r][p >> 1][p & 1].m_error + g_bc7_mode_0_optimal_endpoints[g][p >> 1][p & 1].m_error + g_bc7_mode_0_optimal_endpoints[b][p >> 1][p & 1].m_error;
                                if (err < best_err)
                                {
                                                best_err = err;
                                                best_p = p;
                                }
                }

                const endpoint_err *pEr = &g_bc7_mode_0_optimal_endpoints[r][best_p >> 1][best_p & 1];
                const endpoint_err *pEg = &g_bc7_mode_0_optimal_endpoints[g][best_p >> 1][best_p & 1];
                const endpoint_err *pEb = &g_bc7_mode_0_optimal_endpoints[b][best_p >> 1][best_p & 1];

// #pragma ignore warning(perf)  [ISPC pragma, no-op in C++]
                color_quad_i_set(&pResults->m_low_endpoint, pEr->m_lo, pEg->m_lo, pEb->m_lo, 0);

// #pragma ignore warning(perf)  [ISPC pragma, no-op in C++]
                color_quad_i_set(&pResults->m_high_endpoint, pEr->m_hi, pEg->m_hi, pEb->m_hi, 0);

                pResults->m_pbits[0] = best_p & 1;
                pResults->m_pbits[1] = best_p >> 1;

                for ( uint32_t i = 0; i < num_pixels; i++)
                                pSelectors[i] = BC7E_MODE_0_OPTIMAL_INDEX;

                color_quad_i p;
                
                for ( uint32_t i = 0; i < 3; i++)
                {
                                uint32_t low = ((pResults->m_low_endpoint.m_c[i] << 1) | pResults->m_pbits[0]) << 3;
                                low |= (low >> 5);

                                uint32_t high = ((pResults->m_high_endpoint.m_c[i] << 1) | pResults->m_pbits[1]) << 3;
                                high |= (high >> 5);

                                p.m_c[i] = (low * (64 - g_bc7_weights3[BC7E_MODE_0_OPTIMAL_INDEX]) + high * g_bc7_weights3[BC7E_MODE_0_OPTIMAL_INDEX] + 32) >> 6;
                }
                
                p.m_c[3] = 255;

                uint64_t total_err = 0;
                for ( uint32_t i = 0; i < num_pixels; i++)
                                total_err += compute_color_distance_rgb(&p, &pPixels[i], pParams->m_perceptual, pParams->m_weights);

                pResults->m_best_overall_err = total_err;

                return total_err;
}

static uint64_t pack_mode6_to_one_color(const  color_cell_compressor_params * pParams,  color_cell_compressor_results * pResults, uint32_t r, uint32_t g, uint32_t b, uint32_t a,
                 int * pSelectors, uint32_t num_pixels, const  color_quad_i * pPixels)
{
                uint32_t best_err = UINT_MAX;
                uint32_t best_p = 0;

                for ( uint32_t p = 0; p < 4; p++)
                {
                                 uint32_t hi_p = p >> 1;
                                 uint32_t lo_p = p & 1;
// #pragma ignore warning(perf)  [ISPC pragma, no-op in C++]
                                uint32_t err = g_bc7_mode_6_optimal_endpoints[r][hi_p][lo_p].m_error + g_bc7_mode_6_optimal_endpoints[g][hi_p][lo_p].m_error + g_bc7_mode_6_optimal_endpoints[b][hi_p][lo_p].m_error + g_bc7_mode_6_optimal_endpoints[a][hi_p][lo_p].m_error;
                                if (err < best_err)
                                {
                                                best_err = err;
                                                best_p = p;
                                }
                }

                uint32_t best_hi_p = best_p >> 1;
                uint32_t best_lo_p = best_p & 1;

                const endpoint_err *pEr = &g_bc7_mode_6_optimal_endpoints[r][best_hi_p][best_lo_p];
                const endpoint_err *pEg = &g_bc7_mode_6_optimal_endpoints[g][best_hi_p][best_lo_p];
                const endpoint_err *pEb = &g_bc7_mode_6_optimal_endpoints[b][best_hi_p][best_lo_p];
                const endpoint_err *pEa = &g_bc7_mode_6_optimal_endpoints[a][best_hi_p][best_lo_p];

// #pragma ignore warning(perf)  [ISPC pragma, no-op in C++]
                color_quad_i_set(&pResults->m_low_endpoint, pEr->m_lo, pEg->m_lo, pEb->m_lo, pEa->m_lo);

// #pragma ignore warning(perf)  [ISPC pragma, no-op in C++]
                color_quad_i_set(&pResults->m_high_endpoint, pEr->m_hi, pEg->m_hi, pEb->m_hi, pEa->m_hi);

                pResults->m_pbits[0] = best_lo_p;
                pResults->m_pbits[1] = best_hi_p;

                for ( uint32_t i = 0; i < num_pixels; i++)
                                pSelectors[i] = BC7E_MODE_6_OPTIMAL_INDEX;

                color_quad_i p;
                
                for ( uint32_t i = 0; i < 4; i++)
                {
                                uint32_t low = (pResults->m_low_endpoint.m_c[i] << 1) | pResults->m_pbits[0];
                                uint32_t high = (pResults->m_high_endpoint.m_c[i] << 1) | pResults->m_pbits[1];
                                
                                p.m_c[i] = (low * (64 - g_bc7_weights4[BC7E_MODE_6_OPTIMAL_INDEX]) + high * g_bc7_weights4[BC7E_MODE_6_OPTIMAL_INDEX] + 32) >> 6;
                }

                uint64_t total_err = 0;
                for ( uint32_t i = 0; i < num_pixels; i++)
                                total_err += compute_color_distance_rgba(&p, &pPixels[i], pParams->m_perceptual, pParams->m_weights);

                pResults->m_best_overall_err = total_err;

                return total_err;
}

static uint64_t pack_mode7_to_one_color(const  color_cell_compressor_params * pParams,  color_cell_compressor_results * pResults, uint32_t r, uint32_t g, uint32_t b, uint32_t a,
                 int * pSelectors, uint32_t num_pixels, const  color_quad_i * pPixels)
{
                uint32_t best_err = UINT_MAX;
                uint32_t best_p = 0;

                for ( uint32_t p = 0; p < 4; p++)
                {
                                 uint32_t hi_p = p >> 1;
                                 uint32_t lo_p = p & 1;
// #pragma ignore warning(perf)  [ISPC pragma, no-op in C++]
                                uint32_t err = g_bc7_mode_7_optimal_endpoints[r][hi_p][lo_p].m_error + g_bc7_mode_7_optimal_endpoints[g][hi_p][lo_p].m_error + g_bc7_mode_7_optimal_endpoints[b][hi_p][lo_p].m_error + g_bc7_mode_7_optimal_endpoints[a][hi_p][lo_p].m_error;
                                if (err < best_err)
                                {
                                                best_err = err;
                                                best_p = p;
                                }
                }

                uint32_t best_hi_p = best_p >> 1;
                uint32_t best_lo_p = best_p & 1;

                const endpoint_err *pEr = &g_bc7_mode_7_optimal_endpoints[r][best_hi_p][best_lo_p];
                const endpoint_err *pEg = &g_bc7_mode_7_optimal_endpoints[g][best_hi_p][best_lo_p];
                const endpoint_err *pEb = &g_bc7_mode_7_optimal_endpoints[b][best_hi_p][best_lo_p];
                const endpoint_err *pEa = &g_bc7_mode_7_optimal_endpoints[a][best_hi_p][best_lo_p];

// #pragma ignore warning(perf)  [ISPC pragma, no-op in C++]
                color_quad_i_set(&pResults->m_low_endpoint, pEr->m_lo, pEg->m_lo, pEb->m_lo, pEa->m_lo);

// #pragma ignore warning(perf)  [ISPC pragma, no-op in C++]
                color_quad_i_set(&pResults->m_high_endpoint, pEr->m_hi, pEg->m_hi, pEb->m_hi, pEa->m_hi);

                pResults->m_pbits[0] = best_lo_p;
                pResults->m_pbits[1] = best_hi_p;

                for ( uint32_t i = 0; i < num_pixels; i++)
                                pSelectors[i] = BC7E_MODE_7_OPTIMAL_INDEX;

                color_quad_i p;
                
                for ( uint32_t i = 0; i < 4; i++)
                {
                                uint32_t low = (pResults->m_low_endpoint.m_c[i] << 1) | pResults->m_pbits[0];
                                uint32_t high = (pResults->m_high_endpoint.m_c[i] << 1) | pResults->m_pbits[1];
                                
                                p.m_c[i] = (low * (64 - g_bc7_weights2[BC7E_MODE_7_OPTIMAL_INDEX]) + high * g_bc7_weights2[BC7E_MODE_7_OPTIMAL_INDEX] + 32) >> 6;
                }

                uint64_t total_err = 0;
                for ( uint32_t i = 0; i < num_pixels; i++)
                                total_err += compute_color_distance_rgba(&p, &pPixels[i], pParams->m_perceptual, pParams->m_weights);

                pResults->m_best_overall_err = total_err;

                return total_err;
}

static uint64_t evaluate_solution(const color_quad_i *pLow, const color_quad_i *pHigh, const uint32_t *pbits,
                                                                  const color_cell_compressor_params *pParams, color_cell_compressor_results *pResults, 
                                                                  uint32_t num_pixels, const color_quad_i *pPixels, const int *dup_of)
{
        color_quad_i quantMinColor = *pLow;
        color_quad_i quantMaxColor = *pHigh;
        if (pParams->m_has_pbits)
        {
                uint32_t minPBit, maxPBit;
                if (pParams->m_endpoints_share_pbit) maxPBit = minPBit = pbits[0];
                else { minPBit = pbits[0]; maxPBit = pbits[1]; }
                quantMinColor.m_c[0] = (pLow->m_c[0] << 1) | minPBit; quantMinColor.m_c[1] = (pLow->m_c[1] << 1) | minPBit;
                quantMinColor.m_c[2] = (pLow->m_c[2] << 1) | minPBit; quantMinColor.m_c[3] = (pLow->m_c[3] << 1) | minPBit;
                quantMaxColor.m_c[0] = (pHigh->m_c[0] << 1) | maxPBit; quantMaxColor.m_c[1] = (pHigh->m_c[1] << 1) | maxPBit;
                quantMaxColor.m_c[2] = (pHigh->m_c[2] << 1) | maxPBit; quantMaxColor.m_c[3] = (pHigh->m_c[3] << 1) | maxPBit;
        }
        color_quad_i actualMinColor = scale_color(&quantMinColor, pParams);
        color_quad_i actualMaxColor = scale_color(&quantMaxColor, pParams);
        const uint32_t N = pParams->m_num_selector_weights;
        const uint32_t nc = pParams->m_has_alpha ? 4 : 3;
        float total_errf = 0;
        float wr = pParams->m_weights[0]; float wg = pParams->m_weights[1];
        float wb = pParams->m_weights[2]; float wa = pParams->m_weights[3];

        // Precompute float weighted colors for N=8, N=4, and perceptual modes
        color_quad_f weightedColors[16];
        weightedColors[0].m_c[0] = actualMinColor.m_c[0]; weightedColors[0].m_c[1] = actualMinColor.m_c[1];
        weightedColors[0].m_c[2] = actualMinColor.m_c[2]; weightedColors[0].m_c[3] = actualMinColor.m_c[3];
        weightedColors[N - 1].m_c[0] = actualMaxColor.m_c[0]; weightedColors[N - 1].m_c[1] = actualMaxColor.m_c[1];
        weightedColors[N - 1].m_c[2] = actualMaxColor.m_c[2]; weightedColors[N - 1].m_c[3] = actualMaxColor.m_c[3];
        for (uint32_t i = 1; i < (N - 1); i++)
                for (uint32_t j = 0; j < nc; j++)
                        weightedColors[i].m_c[j] = floor((weightedColors[0].m_c[j] * (64.0f - pParams->m_pSelector_weights[i]) + weightedColors[N - 1].m_c[j] * pParams->m_pSelector_weights[i] + 32) * (1.0f / 64.0f));

        // PIXEL DEDUPLICATION CACHE: dup_of[] computed once in color_cell_compression,
        // passed as const int* parameter. Selector search skips duplicate pixels.
        float pixel_err[16];

        if (!pParams->m_perceptual)
        {
                if (!pParams->m_has_alpha)
                {
                        // =====================================================================
                        // NUCLEAR OPTION: Pure Integer LUT for Mode 6 (N=16)
                        // Eliminates floor(), denom, and float projection. 
                        // Compilers will auto-vectorize this inner loop into SIMD.
                        // =====================================================================
                        if (N == 16)
                        {
                                uint8_t interp_r[16], interp_g[16], interp_b[16];
                                for (uint32_t s = 0; s < 16; s++) {
                                        uint32_t w = pParams->m_pSelector_weights[s];
                                        interp_r[s] = (uint8_t)(((64 - w) * actualMinColor.m_c[0] + w * actualMaxColor.m_c[0] + 32) >> 6);
                                        interp_g[s] = (uint8_t)(((64 - w) * actualMinColor.m_c[1] + w * actualMaxColor.m_c[1] + 32) >> 6);
                                        interp_b[s] = (uint8_t)(((64 - w) * actualMinColor.m_c[2] + w * actualMaxColor.m_c[2] + 32) >> 6);
                                }
#if BC7E_HAVE_SSE2
                                // -----------------------------------------------------------------
                                // SSE2 path: 8 selectors per SIMD group, 2 groups per pixel.
                                // Bit-exact with the scalar path when wr==wg==wb==1.0f.
                                // Tie-breaking (strict <) is preserved by processing groups
                                // in ascending selector order. Uses goto so the scalar
                                // fallback below stays byte-identical to the v2 baseline
                                // (compiler auto-vectorization of the scalar loop is
                                // preserved unchanged).
                                // -----------------------------------------------------------------
                                if (wr == 1.0f && wg == 1.0f && wb == 1.0f)
                                {
                                        const __m128i vr_lo = _mm_setr_epi16(interp_r[0], interp_r[1], interp_r[2], interp_r[3], interp_r[4], interp_r[5], interp_r[6], interp_r[7]);
                                        const __m128i vr_hi = _mm_setr_epi16(interp_r[8], interp_r[9], interp_r[10],interp_r[11],interp_r[12],interp_r[13],interp_r[14],interp_r[15]);
                                        const __m128i vg_lo = _mm_setr_epi16(interp_g[0], interp_g[1], interp_g[2], interp_g[3], interp_g[4], interp_g[5], interp_g[6], interp_g[7]);
                                        const __m128i vg_hi = _mm_setr_epi16(interp_g[8], interp_g[9], interp_g[10],interp_g[11],interp_g[12],interp_g[13],interp_g[14],interp_g[15]);
                                        const __m128i vb_lo = _mm_setr_epi16(interp_b[0], interp_b[1], interp_b[2], interp_b[3], interp_b[4], interp_b[5], interp_b[6], interp_b[7]);
                                        const __m128i vb_hi = _mm_setr_epi16(interp_b[8], interp_b[9], interp_b[10],interp_b[11],interp_b[12],interp_b[13],interp_b[14],interp_b[15]);
                                        const __m128i zero = _mm_setzero_si128();
                                        const __m128i vrs[2] = { vr_lo, vr_hi };
                                        const __m128i vgs[2] = { vg_lo, vg_hi };
                                        const __m128i vbs[2] = { vb_lo, vb_hi };

                                        for (uint32_t i = 0; i < num_pixels; i++) {
                                                if (dup_of[i] >= 0) { pResults->m_pSelectors_temp[i] = pResults->m_pSelectors_temp[dup_of[i]]; total_errf += pixel_err[dup_of[i]]; continue; }
                                                const color_quad_i *pC = &pPixels[i];
                                                const __m128i vpr = _mm_set1_epi16((short)pC->m_c[0]);
                                                const __m128i vpg = _mm_set1_epi16((short)pC->m_c[1]);
                                                const __m128i vpb = _mm_set1_epi16((short)pC->m_c[2]);

                                                uint32_t best_err = UINT32_MAX;
                                                int best_sel = 0;

                                                for (int g = 0; g < 2; g++)
                                                {
                                                        __m128i dr = _mm_sub_epi16(vrs[g], vpr);
                                                        __m128i dg = _mm_sub_epi16(vgs[g], vpg);
                                                        __m128i db = _mm_sub_epi16(vbs[g], vpb);

                                                        __m128i drdg_lo = _mm_unpacklo_epi16(dr, dg);
                                                        __m128i drdg_hi = _mm_unpackhi_epi16(dr, dg);
                                                        __m128i rd_lo = _mm_madd_epi16(drdg_lo, drdg_lo);
                                                        __m128i rd_hi = _mm_madd_epi16(drdg_hi, drdg_hi);

                                                        __m128i db2 = _mm_mullo_epi16(db, db);
                                                        __m128i db2_lo = _mm_unpacklo_epi16(db2, zero);
                                                        __m128i db2_hi = _mm_unpackhi_epi16(db2, zero);

                                                        __m128i err_lo = _mm_add_epi32(rd_lo, db2_lo);
                                                        __m128i err_hi = _mm_add_epi32(rd_hi, db2_hi);

                                                        uint32_t errs[8];
                                                        _mm_storeu_si128((__m128i*)&errs[0], err_lo);
                                                        _mm_storeu_si128((__m128i*)&errs[4], err_hi);

                                                        for (int k = 0; k < 8; k++) {
                                                                if (errs[k] < best_err) { best_err = errs[k]; best_sel = 8*g + k; }
                                                        }
                                                }
                                                pixel_err[i] = (float)best_err;
                                                total_errf += best_err;
                                                pResults->m_pSelectors_temp[i] = best_sel;
                                        }
                                        goto n16_rgb_done;
                                }
#endif
                                for (uint32_t i = 0; i < num_pixels; i++) {
     if (dup_of[i] >= 0) { pResults->m_pSelectors_temp[i] = pResults->m_pSelectors_temp[dup_of[i]]; total_errf += pixel_err[dup_of[i]]; continue; }
                                        const color_quad_i *pC = &pPixels[i];
                                        uint32_t best_err = UINT32_MAX;
                                        int best_sel = 0;
                                        for (uint32_t s = 0; s < 16; s++) {
                                                int dr = (int)interp_r[s] - pC->m_c[0];
                                                int dg = (int)interp_g[s] - pC->m_c[1];
                                                int db = (int)interp_b[s] - pC->m_c[2];
                                                uint32_t err = (uint32_t)(wr*(dr*dr) + wg*(dg*dg) + wb*(db*db));
                                                if (err < best_err) { best_err = err; best_sel = s; }
                                        }
     pixel_err[i] = (float)best_err;
                                        total_errf += best_err;
                                        pResults->m_pSelectors_temp[i] = best_sel;
                                }
                                n16_rgb_done: ;
                        }
                        else if (N == 8)
                        {
                                for (uint32_t i = 0; i < num_pixels; i++) {
     if (dup_of[i] >= 0) { pResults->m_pSelectors_temp[i] = pResults->m_pSelectors_temp[dup_of[i]]; total_errf += pixel_err[dup_of[i]]; continue; }
                                        float pr = (float)pPixels[i].m_c[0]; float pg = (float)pPixels[i].m_c[1]; float pb = (float)pPixels[i].m_c[2];
                                        float best_err; int best_sel;
                                        {
                                                float dr0 = weightedColors[0].m_c[0] - pr; float dg0 = weightedColors[0].m_c[1] - pg; float db0 = weightedColors[0].m_c[2] - pb; float err0 = wr * dr0 * dr0 + wg * dg0 * dg0 + wb * db0 * db0;
                                                float dr1 = weightedColors[1].m_c[0] - pr; float dg1 = weightedColors[1].m_c[1] - pg; float db1 = weightedColors[1].m_c[2] - pb; float err1 = wr * dr1 * dr1 + wg * dg1 * dg1 + wb * db1 * db1;
                                                float dr2 = weightedColors[2].m_c[0] - pr; float dg2 = weightedColors[2].m_c[1] - pg; float db2 = weightedColors[2].m_c[2] - pb; float err2 = wr * dr2 * dr2 + wg * dg2 * dg2 + wb * db2 * db2;
                                                float dr3 = weightedColors[3].m_c[0] - pr; float dg3 = weightedColors[3].m_c[1] - pg; float db3 = weightedColors[3].m_c[2] - pb; float err3 = wr * dr3 * dr3 + wg * dg3 * dg3 + wb * db3 * db3;
                                                best_err = min(min(min(err0, err1), err2), err3); best_sel = ispc_select(best_err == err1, 1, 0); best_sel = ispc_select(best_err == err2, 2, best_sel); best_sel = ispc_select(best_err == err3, 3, best_sel);
                                        }
                                        {
                                                float dr0 = weightedColors[4].m_c[0] - pr; float dg0 = weightedColors[4].m_c[1] - pg; float db0 = weightedColors[4].m_c[2] - pb; float err0 = wr * dr0 * dr0 + wg * dg0 * dg0 + wb * db0 * db0;
                                                float dr1 = weightedColors[5].m_c[0] - pr; float dg1 = weightedColors[5].m_c[1] - pg; float db1 = weightedColors[5].m_c[2] - pb; float err1 = wr * dr1 * dr1 + wg * dg1 * dg1 + wb * db1 * db1;
                                                float dr2 = weightedColors[6].m_c[0] - pr; float dg2 = weightedColors[6].m_c[1] - pg; float db2 = weightedColors[6].m_c[2] - pb; float err2 = wr * dr2 * dr2 + wg * dg2 * dg2 + wb * db2 * db2;
                                                float dr3 = weightedColors[7].m_c[0] - pr; float dg3 = weightedColors[7].m_c[1] - pg; float db3 = weightedColors[7].m_c[2] - pb; float err3 = wr * dr3 * dr3 + wg * dg3 * dg3 + wb * db3 * db3;
                                                best_err = min(best_err, min(min(min(err0, err1), err2), err3)); best_sel = ispc_select(best_err == err0, 4, best_sel); best_sel = ispc_select(best_err == err1, 5, best_sel); best_sel = ispc_select(best_err == err2, 6, best_sel); best_sel = ispc_select(best_err == err3, 7, best_sel);
                                        }
     pixel_err[i] = (float)best_err;
                                        total_errf += best_err; pResults->m_pSelectors_temp[i] = best_sel;
                                }
                        }
                        else // N == 4
                        {
                                for (uint32_t i = 0; i < num_pixels; i++) {
     if (dup_of[i] >= 0) { pResults->m_pSelectors_temp[i] = pResults->m_pSelectors_temp[dup_of[i]]; total_errf += pixel_err[dup_of[i]]; continue; }
                                        float pr = (float)pPixels[i].m_c[0]; float pg = (float)pPixels[i].m_c[1]; float pb = (float)pPixels[i].m_c[2];
                                        float dr0 = weightedColors[0].m_c[0] - pr; float dg0 = weightedColors[0].m_c[1] - pg; float db0 = weightedColors[0].m_c[2] - pb; float err0 = wr * dr0 * dr0 + wg * dg0 * dg0 + wb * db0 * db0;
                                        float dr1 = weightedColors[1].m_c[0] - pr; float dg1 = weightedColors[1].m_c[1] - pg; float db1 = weightedColors[1].m_c[2] - pb; float err1 = wr * dr1 * dr1 + wg * dg1 * dg1 + wb * db1 * db1;
                                        float dr2 = weightedColors[2].m_c[0] - pr; float dg2 = weightedColors[2].m_c[1] - pg; float db2 = weightedColors[2].m_c[2] - pb; float err2 = wr * dr2 * dr2 + wg * dg2 * dg2 + wb * db2 * db2;
                                        float dr3 = weightedColors[3].m_c[0] - pr; float dg3 = weightedColors[3].m_c[1] - pg; float db3 = weightedColors[3].m_c[2] - pb; float err3 = wr * dr3 * dr3 + wg * dg3 * dg3 + wb * db3 * db3;
                                        float best_err = min(min(min(err0, err1), err2), err3); int best_sel = ispc_select(best_err == err1, 1, 0); best_sel = ispc_select(best_err == err2, 2, best_sel); best_sel = ispc_select(best_err == err3, 3, best_sel);
     pixel_err[i] = (float)best_err;
                                        total_errf += best_err; pResults->m_pSelectors_temp[i] = best_sel;
                                }
                        }
                }
                else // has_alpha
                {
                        // =====================================================================
                        // NUCLEAR OPTION: Pure Integer LUT for Mode 6 (N=16) with Alpha
                        // =====================================================================
                        if (N == 16)
                        {
                                uint8_t interp_r[16], interp_g[16], interp_b[16], interp_a[16];
                                for (uint32_t s = 0; s < 16; s++) {
                                        uint32_t w = pParams->m_pSelector_weights[s];
                                        interp_r[s] = (uint8_t)(((64 - w) * actualMinColor.m_c[0] + w * actualMaxColor.m_c[0] + 32) >> 6);
                                        interp_g[s] = (uint8_t)(((64 - w) * actualMinColor.m_c[1] + w * actualMaxColor.m_c[1] + 32) >> 6);
                                        interp_b[s] = (uint8_t)(((64 - w) * actualMinColor.m_c[2] + w * actualMaxColor.m_c[2] + 32) >> 6);
                                        interp_a[s] = (uint8_t)(((64 - w) * actualMinColor.m_c[3] + w * actualMaxColor.m_c[3] + 32) >> 6);
                                }
#if BC7E_HAVE_SSE2
                                // -----------------------------------------------------------------
                                // SSE2 path for RGBA N=16. Bit-exact with the scalar path when
                                // wr==wg==wb==wa==1.0f. Uses two _mm_madd_epi16 calls per group
                                // to get dr²+dg² and db²+da² per lane, then sums them.
                                // -----------------------------------------------------------------
                                if (wr == 1.0f && wg == 1.0f && wb == 1.0f && wa == 1.0f)
                                {
                                        const __m128i vr_lo = _mm_setr_epi16(interp_r[0], interp_r[1], interp_r[2], interp_r[3], interp_r[4], interp_r[5], interp_r[6], interp_r[7]);
                                        const __m128i vr_hi = _mm_setr_epi16(interp_r[8], interp_r[9], interp_r[10],interp_r[11],interp_r[12],interp_r[13],interp_r[14],interp_r[15]);
                                        const __m128i vg_lo = _mm_setr_epi16(interp_g[0], interp_g[1], interp_g[2], interp_g[3], interp_g[4], interp_g[5], interp_g[6], interp_g[7]);
                                        const __m128i vg_hi = _mm_setr_epi16(interp_g[8], interp_g[9], interp_g[10],interp_g[11],interp_g[12],interp_g[13],interp_g[14],interp_g[15]);
                                        const __m128i vb_lo = _mm_setr_epi16(interp_b[0], interp_b[1], interp_b[2], interp_b[3], interp_b[4], interp_b[5], interp_b[6], interp_b[7]);
                                        const __m128i vb_hi = _mm_setr_epi16(interp_b[8], interp_b[9], interp_b[10],interp_b[11],interp_b[12],interp_b[13],interp_b[14],interp_b[15]);
                                        const __m128i va_lo = _mm_setr_epi16(interp_a[0], interp_a[1], interp_a[2], interp_a[3], interp_a[4], interp_a[5], interp_a[6], interp_a[7]);
                                        const __m128i va_hi = _mm_setr_epi16(interp_a[8], interp_a[9], interp_a[10],interp_a[11],interp_a[12],interp_a[13],interp_a[14],interp_a[15]);
                                        const __m128i vrs[2] = { vr_lo, vr_hi };
                                        const __m128i vgs[2] = { vg_lo, vg_hi };
                                        const __m128i vbs[2] = { vb_lo, vb_hi };
                                        const __m128i vas[2] = { va_lo, va_hi };

                                        for (uint32_t i = 0; i < num_pixels; i++) {
                                                if (dup_of[i] >= 0) { pResults->m_pSelectors_temp[i] = pResults->m_pSelectors_temp[dup_of[i]]; total_errf += pixel_err[dup_of[i]]; continue; }
                                                const color_quad_i *pC = &pPixels[i];
                                                const __m128i vpr = _mm_set1_epi16((short)pC->m_c[0]);
                                                const __m128i vpg = _mm_set1_epi16((short)pC->m_c[1]);
                                                const __m128i vpb = _mm_set1_epi16((short)pC->m_c[2]);
                                                const __m128i vpa = _mm_set1_epi16((short)pC->m_c[3]);

                                                uint32_t best_err = UINT32_MAX;
                                                int best_sel = 0;

                                                for (int g = 0; g < 2; g++)
                                                {
                                                        __m128i dr = _mm_sub_epi16(vrs[g], vpr);
                                                        __m128i dg = _mm_sub_epi16(vgs[g], vpg);
                                                        __m128i db = _mm_sub_epi16(vbs[g], vpb);
                                                        __m128i da = _mm_sub_epi16(vas[g], vpa);

                                                        __m128i drdg_lo = _mm_unpacklo_epi16(dr, dg);
                                                        __m128i drdg_hi = _mm_unpackhi_epi16(dr, dg);
                                                        __m128i rd_lo = _mm_madd_epi16(drdg_lo, drdg_lo);
                                                        __m128i rd_hi = _mm_madd_epi16(drdg_hi, drdg_hi);

                                                        __m128i dbda_lo = _mm_unpacklo_epi16(db, da);
                                                        __m128i dbda_hi = _mm_unpackhi_epi16(db, da);
                                                        __m128i bd_lo = _mm_madd_epi16(dbda_lo, dbda_lo);
                                                        __m128i bd_hi = _mm_madd_epi16(dbda_hi, dbda_hi);

                                                        __m128i err_lo = _mm_add_epi32(rd_lo, bd_lo);
                                                        __m128i err_hi = _mm_add_epi32(rd_hi, bd_hi);

                                                        uint32_t errs[8];
                                                        _mm_storeu_si128((__m128i*)&errs[0], err_lo);
                                                        _mm_storeu_si128((__m128i*)&errs[4], err_hi);

                                                        for (int k = 0; k < 8; k++) {
                                                                if (errs[k] < best_err) { best_err = errs[k]; best_sel = 8*g + k; }
                                                        }
                                                }
                                                pixel_err[i] = (float)best_err;
                                                total_errf += best_err;
                                                pResults->m_pSelectors_temp[i] = best_sel;
                                        }
                                        goto n16_rgba_done;
                                }
#endif
                                for (uint32_t i = 0; i < num_pixels; i++) {
     if (dup_of[i] >= 0) { pResults->m_pSelectors_temp[i] = pResults->m_pSelectors_temp[dup_of[i]]; total_errf += pixel_err[dup_of[i]]; continue; }
                                        const color_quad_i *pC = &pPixels[i];
                                        uint32_t best_err = UINT32_MAX;
                                        int best_sel = 0;
                                        for (uint32_t s = 0; s < 16; s++) {
                                                int dr = (int)interp_r[s] - pC->m_c[0];
                                                int dg = (int)interp_g[s] - pC->m_c[1];
                                                int db = (int)interp_b[s] - pC->m_c[2];
                                                int da = (int)interp_a[s] - pC->m_c[3];
                                                int err = wr*dr*dr + wg*dg*dg + wb*db*db + wa*da*da;
                                                if (err < best_err) { best_err = err; best_sel = (int)s; }
                                        }
     pixel_err[i] = (float)best_err;
                                        total_errf += best_err;
                                        pResults->m_pSelectors_temp[i] = best_sel;
                                }
                                n16_rgba_done: ;
                        }
                        else if (N == 8)
                        {
                                for (uint32_t i = 0; i < num_pixels; i++) {
     if (dup_of[i] >= 0) { pResults->m_pSelectors_temp[i] = pResults->m_pSelectors_temp[dup_of[i]]; total_errf += pixel_err[dup_of[i]]; continue; }
                                        float pr = (float)pPixels[i].m_c[0]; float pg = (float)pPixels[i].m_c[1]; float pb = (float)pPixels[i].m_c[2]; float pa = (float)pPixels[i].m_c[3];
                                        float best_err; int best_sel;
                                        {
                                                float dr0 = weightedColors[0].m_c[0] - pr; float dg0 = weightedColors[0].m_c[1] - pg; float db0 = weightedColors[0].m_c[2] - pb; float da0 = weightedColors[0].m_c[3] - pa; float err0 = wr * dr0 * dr0 + wg * dg0 * dg0 + wb * db0 * db0 + wa * da0 * da0;
                                                float dr1 = weightedColors[1].m_c[0] - pr; float dg1 = weightedColors[1].m_c[1] - pg; float db1 = weightedColors[1].m_c[2] - pb; float da1 = weightedColors[1].m_c[3] - pa; float err1 = wr * dr1 * dr1 + wg * dg1 * dg1 + wb * db1 * db1 + wa * da1 * da1;
                                                float dr2 = weightedColors[2].m_c[0] - pr; float dg2 = weightedColors[2].m_c[1] - pg; float db2 = weightedColors[2].m_c[2] - pb; float da2 = weightedColors[2].m_c[3] - pa; float err2 = wr * dr2 * dr2 + wg * dg2 * dg2 + wb * db2 * db2 + wa * da2 * da2;
                                                float dr3 = weightedColors[3].m_c[0] - pr; float dg3 = weightedColors[3].m_c[1] - pg; float db3 = weightedColors[3].m_c[2] - pb; float da3 = weightedColors[3].m_c[3] - pa; float err3 = wr * dr3 * dr3 + wg * dg3 * dg3 + wb * db3 * db3 + wa * da3 * da3;
                                                best_err = min(min(min(err0, err1), err2), err3); best_sel = ispc_select(best_err == err1, 1, 0); best_sel = ispc_select(best_err == err2, 2, best_sel); best_sel = ispc_select(best_err == err3, 3, best_sel);
                                        }
                                        {
                                                float dr0 = weightedColors[4].m_c[0] - pr; float dg0 = weightedColors[4].m_c[1] - pg; float db0 = weightedColors[4].m_c[2] - pb; float da0 = weightedColors[4].m_c[3] - pa; float err0 = wr * dr0 * dr0 + wg * dg0 * dg0 + wb * db0 * db0 + wa * da0 * da0;
                                                float dr1 = weightedColors[5].m_c[0] - pr; float dg1 = weightedColors[5].m_c[1] - pg; float db1 = weightedColors[5].m_c[2] - pb; float da1 = weightedColors[5].m_c[3] - pa; float err1 = wr * dr1 * dr1 + wg * dg1 * dg1 + wb * db1 * db1 + wa * da1 * da1;
                                                float dr2 = weightedColors[6].m_c[0] - pr; float dg2 = weightedColors[6].m_c[1] - pg; float db2 = weightedColors[6].m_c[2] - pb; float da2 = weightedColors[6].m_c[3] - pa; float err2 = wr * dr2 * dr2 + wg * dg2 * dg2 + wb * db2 * db2 + wa * da2 * da2;
                                                float dr3 = weightedColors[7].m_c[0] - pr; float dg3 = weightedColors[7].m_c[1] - pg; float db3 = weightedColors[7].m_c[2] - pb; float da3 = weightedColors[7].m_c[3] - pa; float err3 = wr * dr3 * dr3 + wg * dg3 * dg3 + wb * db3 * db3 + wa * da3 * da3;
                                                best_err = min(best_err, min(min(min(err0, err1), err2), err3)); best_sel = ispc_select(best_err == err0, 4, best_sel); best_sel = ispc_select(best_err == err1, 5, best_sel); best_sel = ispc_select(best_err == err2, 6, best_sel); best_sel = ispc_select(best_err == err3, 7, best_sel);
                                        }
     pixel_err[i] = (float)best_err;
                                        total_errf += best_err; pResults->m_pSelectors_temp[i] = best_sel;
                                }
                        }
                        else // N == 4
                        {
                                for (uint32_t i = 0; i < num_pixels; i++) {
     if (dup_of[i] >= 0) { pResults->m_pSelectors_temp[i] = pResults->m_pSelectors_temp[dup_of[i]]; total_errf += pixel_err[dup_of[i]]; continue; }
                                        float pr = (float)pPixels[i].m_c[0]; float pg = (float)pPixels[i].m_c[1]; float pb = (float)pPixels[i].m_c[2]; float pa = (float)pPixels[i].m_c[3];
                                        float dr0 = weightedColors[0].m_c[0] - pr; float dg0 = weightedColors[0].m_c[1] - pg; float db0 = weightedColors[0].m_c[2] - pb; float da0 = weightedColors[0].m_c[3] - pa; float err0 = wr * dr0 * dr0 + wg * dg0 * dg0 + wb * db0 * db0 + wa * da0 * da0;
                                        float dr1 = weightedColors[1].m_c[0] - pr; float dg1 = weightedColors[1].m_c[1] - pg; float db1 = weightedColors[1].m_c[2] - pb; float da1 = weightedColors[1].m_c[3] - pa; float err1 = wr * dr1 * dr1 + wg * dg1 * dg1 + wb * db1 * db1 + wa * da1 * da1;
                                        float dr2 = weightedColors[2].m_c[0] - pr; float dg2 = weightedColors[2].m_c[1] - pg; float db2 = weightedColors[2].m_c[2] - pb; float da2 = weightedColors[2].m_c[3] - pa; float err2 = wr * dr2 * dr2 + wg * dg2 * dg2 + wb * db2 * db2 + wa * da2 * da2;
                                        float dr3 = weightedColors[3].m_c[0] - pr; float dg3 = weightedColors[3].m_c[1] - pg; float db3 = weightedColors[3].m_c[2] - pb; float da3 = weightedColors[3].m_c[3] - pa; float err3 = wr * dr3 * dr3 + wg * dg3 * dg3 + wb * db3 * db3 + wa * da3 * da3;
                                        float best_err = min(min(min(err0, err1), err2), err3); int best_sel = ispc_select(best_err == err1, 1, 0); best_sel = ispc_select(best_err == err2, 2, best_sel); best_sel = ispc_select(best_err == err3, 3, best_sel);
     pixel_err[i] = (float)best_err;
                                        total_errf += best_err; pResults->m_pSelectors_temp[i] = best_sel;
                                }
                        }
                }
        }
        else // perceptual
        {
                wg *= pr_weight; wb *= pb_weight;
                float weightedColorsY[16], weightedColorsCr[16], weightedColorsCb[16];
                for (uint32_t i = 0; i < N; i++) {
                        float r = weightedColors[i].m_c[0]; float g = weightedColors[i].m_c[1]; float b = weightedColors[i].m_c[2];
                        float y = r * .2126f + g * .7152f + b * .0722f;
                        weightedColorsY[i] = y; weightedColorsCr[i] = r - y; weightedColorsCb[i] = b - y;
                }
                if (pParams->m_has_alpha) {
                        for (uint32_t i = 0; i < num_pixels; i++) {
    if (dup_of[i] >= 0) { pResults->m_pSelectors_temp[i] = pResults->m_pSelectors_temp[dup_of[i]]; total_errf += pixel_err[dup_of[i]]; continue; }
                                float r = pPixels[i].m_c[0]; float g = pPixels[i].m_c[1]; float b = pPixels[i].m_c[2]; float a = pPixels[i].m_c[3];
                                float y = r * .2126f + g * .7152f + b * .0722f; float cr = r - y; float cb = b - y;
                                // GLOW FIX: Premultiplied alpha error weighting.
                                // Weight RGB error by (alpha/255)² so transparent pixels' RGB
                                // (often garbage from the image pipeline) doesn't poison endpoint/
                                // selector optimization. This eliminates blocky-glow seams in
                                // semi-transparent gradient regions. Opaque pixels (alpha=255)
                                // get weight=1, so opaque blocks are completely unaffected.
                                float aw = a * (1.0f / 255.0f);
                                float best_err = 1e+10f; int32_t best_sel;
                                for (uint32_t j = 0; j < N; j++) {
                                        float dl = y - weightedColorsY[j]; float dcr = cr - weightedColorsCr[j]; float dcb = cb - weightedColorsCb[j]; float da = a - weightedColors[j].m_c[3];
                                        float rgb_err = (wr * dl * dl) + (wg * dcr * dcr) + (wb * dcb * dcb);
                                        float err = aw * rgb_err + (wa * da * da);
                                        if (err < best_err) { best_err = err; best_sel = j; }
                                }
    pixel_err[i] = (float)best_err;
                                total_errf += best_err; pResults->m_pSelectors_temp[i] = best_sel;
                        }
                } else {
                        for (uint32_t i = 0; i < num_pixels; i++) {
    if (dup_of[i] >= 0) { pResults->m_pSelectors_temp[i] = pResults->m_pSelectors_temp[dup_of[i]]; total_errf += pixel_err[dup_of[i]]; continue; }
                                float r = pPixels[i].m_c[0]; float g = pPixels[i].m_c[1]; float b = pPixels[i].m_c[2];
                                float y = r * .2126f + g * .7152f + b * .0722f; float cr = r - y; float cb = b - y;
                                float best_err = 1e+10f; int32_t best_sel;
                                for (uint32_t j = 0; j < N; j++) {
                                        float dl = y - weightedColorsY[j]; float dcr = cr - weightedColorsCr[j]; float dcb = cb - weightedColorsCb[j];
                                        float err = (wr * dl * dl) + (wg * dcr * dcr) + (wb * dcb * dcb);
                                        if (err < best_err) { best_err = err; best_sel = j; }
                                }
    pixel_err[i] = (float)best_err;
                                total_errf += best_err; pResults->m_pSelectors_temp[i] = best_sel;
                        }
                }
        }

        uint64_t total_err = (int64_t)total_errf;
        if (total_err < pResults->m_best_overall_err)
        {
                pResults->m_best_overall_err = total_err;
                pResults->m_low_endpoint = *pLow; pResults->m_high_endpoint = *pHigh;
                pResults->m_pbits[0] = pbits[0]; pResults->m_pbits[1] = pbits[1];
                for (uint32_t i = 0; i < num_pixels; i++) pResults->m_pSelectors[i] = pResults->m_pSelectors_temp[i];
        }
        return total_err;
}

static void fixDegenerateEndpoints( uint32_t mode,  color_quad_i * pTrialMinColor,  color_quad_i * pTrialMaxColor, const  vec4F * pXl, const  vec4F * pXh,  uint32_t iscale)
{
                if ((mode == 1) || (mode == 4)) // also mode 2
                {
                                // fix degenerate case where the input collapses to a single colorspace voxel, and we loose all freedom (test with grayscale ramps)
                                for ( uint32_t i = 0; i < 3; i++)
                                {
                                                if (pTrialMinColor->m_c[i] == pTrialMaxColor->m_c[i])
                                                {
                                                                if (abs(pXl->m_c[i] - pXh->m_c[i]) > 0.0f)
                                                                {
                                                                                if (pTrialMinColor->m_c[i] > (iscale >> 1))
                                                                                {
                                                                                                if (pTrialMinColor->m_c[i] > 0)
                                                                                                                pTrialMinColor->m_c[i]--;
                                                                                                else
                                                                                                                if (pTrialMaxColor->m_c[i] < iscale)
                                                                                                                                pTrialMaxColor->m_c[i]++;
                                                                                }
                                                                                else
                                                                                {
                                                                                                if (pTrialMaxColor->m_c[i] < iscale)
                                                                                                                pTrialMaxColor->m_c[i]++;
                                                                                                else if (pTrialMinColor->m_c[i] > 0)
                                                                                                                pTrialMinColor->m_c[i]--;
                                                                                }

                                                                                if (mode == 4)
                                                                                {
                                                                                                if (pTrialMinColor->m_c[i] > (iscale >> 1))
                                                                                                {
                                                                                                                if (pTrialMaxColor->m_c[i] < iscale)
                                                                                                                                pTrialMaxColor->m_c[i]++;
                                                                                                                else if (pTrialMinColor->m_c[i] > 0)
                                                                                                                                pTrialMinColor->m_c[i]--;
                                                                                                }
                                                                                                else
                                                                                                {
                                                                                                                if (pTrialMinColor->m_c[i] > 0)
                                                                                                                                pTrialMinColor->m_c[i]--;
                                                                                                                else if (pTrialMaxColor->m_c[i] < iscale)
                                                                                                                                pTrialMaxColor->m_c[i]++;
                                                                                                }
                                                                                }
                                                                }
                                                }
                                }
                }
}

static uint64_t find_optimal_solution( uint32_t mode,  vec4F * pXl,  vec4F * pXh, const  color_cell_compressor_params * pParams,  color_cell_compressor_results * pResults, 
         bool pbit_search, uint32_t num_pixels, const  color_quad_i * pPixels, const int *pDupOf)
{
        vec4F xl = *pXl;
        vec4F xh = *pXh;

        vec4F_saturate_in_place(&xl);
        vec4F_saturate_in_place(&xh);
                
        if (pParams->m_has_pbits)
        {
                if (pbit_search)
                {
                        // compensated rounding+pbit search
                        const  int iscalep = (1 << (pParams->m_comp_bits + 1)) - 1;
                        const  float scalep = (float)iscalep;

                        const  int32_t totalComps = pParams->m_has_alpha ? 4 : 3;

                        if (!pParams->m_endpoints_share_pbit)
                        {
                                color_quad_i lo[2], hi[2];
                                                                
                                for ( int p = 0; p < 2; p++)
                                {
                                        color_quad_i xMinColor, xMaxColor;

                                        // Notes: The pbit controls which quantization intervals are selected.
                                        // total_levels=2^(comp_bits+1), where comp_bits=4 for mode 0, etc.
                                        // pbit 0: v=(b*2)/(total_levels-1), pbit 1: v=(b*2+1)/(total_levels-1) where b is the component bin from [0,total_levels/2-1] and v is the [0,1] component value
                                        // rearranging you get for pbit 0: b=floor(v*(total_levels-1)/2+.5)
                                        // rearranging you get for pbit 1: b=floor((v*(total_levels-1)-1)/2+.5)
                                        for ( uint32_t c = 0; c < 4; c++)
                                        {
                                                xMinColor.m_c[c] = (int)((xl.m_c[c] * scalep - p) / 2.0f + .5f) * 2 + p;
                                                xMinColor.m_c[c] = ispc_clamp(xMinColor.m_c[c], p, iscalep - 1 + p);

                                                xMaxColor.m_c[c] = (int)((xh.m_c[c] * scalep - p) / 2.0f + .5f) * 2 + p;
                                                xMaxColor.m_c[c] = ispc_clamp(xMaxColor.m_c[c], p, iscalep - 1 + p);
                                        }
                                                                                                                                                                
                                        lo[p] = xMinColor;
                                        hi[p] = xMaxColor;

                                        for ( int c = 0; c < 4; c++)
                                        {
                                                lo[p].m_c[c] >>= 1;
                                                hi[p].m_c[c] >>= 1;
                                        }
                                }

                                fixDegenerateEndpoints(mode, &lo[0], &hi[0], &xl, &xh, iscalep >> 1);
                                fixDegenerateEndpoints(mode, &lo[1], &hi[1], &xl, &xh, iscalep >> 1);

                                uint32_t pbits[2];
                                
                                // ==========================================
                                // MODE 6 LUT P-BIT PREDICTION (INDEPENDENT PBITS)
                                // Predict the best P-bits using just the endpoints to avoid 3 extra evaluate_solution calls.
                                // ==========================================
                                if (mode == 6)
                                {
                                        uint64_t best_p_err = UINT64_MAX;
                                        uint32_t best_p0 = 0, best_p1 = 0;

                                        for (int p0 = 0; p0 < 2; p0++) {
                                                for (int p1 = 0; p1 < 2; p1++) {
                                                        uint64_t err = 
                                                                g_bc7_mode6_lut[lo[0].m_c[0]][p1][p0].err_r + g_bc7_mode6_lut[lo[0].m_c[1]][p1][p0].err_g + 
                                                                g_bc7_mode6_lut[lo[0].m_c[2]][p1][p0].err_b + g_bc7_mode6_lut[lo[0].m_c[3]][p1][p0].err_a +
                                                                g_bc7_mode6_lut[hi[0].m_c[0]][p1][p0].err_r + g_bc7_mode6_lut[hi[0].m_c[1]][p1][p0].err_g + 
                                                                g_bc7_mode6_lut[hi[0].m_c[2]][p1][p0].err_b + g_bc7_mode6_lut[hi[0].m_c[3]][p1][p0].err_a +
                                                                g_bc7_mode6_lut[lo[1].m_c[0]][p1][p0].err_r + g_bc7_mode6_lut[lo[1].m_c[1]][p1][p0].err_g + 
                                                                g_bc7_mode6_lut[lo[1].m_c[2]][p1][p0].err_b + g_bc7_mode6_lut[lo[1].m_c[3]][p1][p0].err_a +
                                                                g_bc7_mode6_lut[hi[1].m_c[0]][p1][p0].err_r + g_bc7_mode6_lut[hi[1].m_c[1]][p1][p0].err_g + 
                                                                g_bc7_mode6_lut[hi[1].m_c[2]][p1][p0].err_b + g_bc7_mode6_lut[hi[1].m_c[3]][p1][p0].err_a;

                                                        if (err < best_p_err) {
                                                                best_p_err = err;
                                                                best_p0 = p0;
                                                                best_p1 = p1;
                                                        }
                                                }
                                        }
                                        
                                        pbits[0] = best_p0;
                                        pbits[1] = best_p1;
                                        evaluate_solution(&lo[best_p0], &hi[best_p1], pbits, pParams, pResults, num_pixels, pPixels, pDupOf);
                                }
                                else if (mode == 0)
                                // ==========================================
                                // MODE 0 P-BIT PREDICTION VIA QUANTIZATION ERROR LUT
                                // ==========================================
                                {
                                        uint64_t err_lo0 = 0, err_lo1 = 0, err_hi0 = 0, err_hi1 = 0;
                                        const uint32_t ncomps = pParams->m_has_alpha ? 4 : 3;
                                        for (uint32_t c = 0; c < ncomps; c++) {
                                                uint32_t lo_val = (uint32_t)(xl.m_c[c] * 255.0f + 0.5f); if (lo_val > 255) lo_val = 255;
                                                uint32_t hi_val = (uint32_t)(xh.m_c[c] * 255.0f + 0.5f); if (hi_val > 255) hi_val = 255;
                                                err_lo0 += g_bc7_mode0_lut[lo_val][0];
                                                err_lo1 += g_bc7_mode0_lut[lo_val][1];
                                                err_hi0 += g_bc7_mode0_lut[hi_val][0];
                                                err_hi1 += g_bc7_mode0_lut[hi_val][1];
                                        }
                                        uint32_t best_p0 = (err_lo0 <= err_lo1) ? 0 : 1;
                                        uint32_t best_p1 = (err_hi0 <= err_hi1) ? 0 : 1;
                                        pbits[0] = best_p0;
                                        pbits[1] = best_p1;
                                        evaluate_solution(&lo[best_p0], &hi[best_p1], pbits, pParams, pResults, num_pixels, pPixels, pDupOf);
                                }
                                else
                                // ==========================================
                                // ORIGINAL CODE (Non-Mode 6, Non-Mode 0)
                                // ==========================================
                                {
                                    pbits[0] = 0; pbits[1] = 0;
                                    evaluate_solution(&lo[0], &hi[0], pbits, pParams, pResults, num_pixels, pPixels, pDupOf);

                                    pbits[0] = 0; pbits[1] = 1;
                                    evaluate_solution(&lo[0], &hi[1], pbits, pParams, pResults, num_pixels, pPixels, pDupOf);

                                    pbits[0] = 1; pbits[1] = 0;
                                    evaluate_solution(&lo[1], &hi[0], pbits, pParams, pResults, num_pixels, pPixels, pDupOf);
                                    
                                    pbits[0] = 1; pbits[1] = 1;
                                    evaluate_solution(&lo[1], &hi[1], pbits, pParams, pResults, num_pixels, pPixels, pDupOf);
                                }
                        }
                        else
                        {
                                // Endpoints share pbits
                                color_quad_i lo[2], hi[2];

                                for ( int p = 0; p < 2; p++)
                                {
                                        color_quad_i xMinColor, xMaxColor;
                                                                
                                        for ( uint32_t c = 0; c < 4; c++)
                                        {
                                                xMinColor.m_c[c] = (int)((xl.m_c[c] * scalep - p) / 2.0f + .5f) * 2 + p;
                                                xMinColor.m_c[c] = ispc_clamp(xMinColor.m_c[c], p, iscalep - 1 + p);

                                                xMaxColor.m_c[c] = (int)((xh.m_c[c] * scalep - p) / 2.0f + .5f) * 2 + p;
                                                xMaxColor.m_c[c] = ispc_clamp(xMaxColor.m_c[c], p, iscalep - 1 + p);
                                        }
                                                                                
                                        lo[p] = xMinColor;
                                        hi[p] = xMaxColor;

                                        for ( int c = 0; c < 4; c++)
                                        {
                                                lo[p].m_c[c] >>= 1;
                                                hi[p].m_c[c] >>= 1;
                                        }
                                }

                                fixDegenerateEndpoints(mode, &lo[0], &hi[0], &xl, &xh, iscalep >> 1);
                                fixDegenerateEndpoints(mode, &lo[1], &hi[1], &xl, &xh, iscalep >> 1);
                                
                                uint32_t pbits[2];
                                
                                // ==========================================
                                // MODE 6 LUT P-BIT PREDICTION (SHARED PBITS)
                                // ==========================================
                                if (mode == 6)
                                {
                                        uint64_t best_p_err = UINT64_MAX;
                                        uint32_t best_p = 0;

                                        for (int p = 0; p < 2; p++) {
                                                        uint64_t err = 
                                                                g_bc7_mode6_lut[lo[p].m_c[0]][p][p].err_r + g_bc7_mode6_lut[lo[p].m_c[1]][p][p].err_g + 
                                                                g_bc7_mode6_lut[lo[p].m_c[2]][p][p].err_b + g_bc7_mode6_lut[lo[p].m_c[3]][p][p].err_a +
                                                                g_bc7_mode6_lut[hi[p].m_c[0]][p][p].err_r + g_bc7_mode6_lut[hi[p].m_c[1]][p][p].err_g + 
                                                                g_bc7_mode6_lut[hi[p].m_c[2]][p][p].err_b + g_bc7_mode6_lut[hi[p].m_c[3]][p][p].err_a;

                                                        if (err < best_p_err) {
                                                                best_p_err = err;
                                                                best_p = p;
                                                        }
                                        }
                                        
                                        pbits[0] = best_p;
                                        pbits[1] = best_p;
                                        evaluate_solution(&lo[best_p], &hi[best_p], pbits, pParams, pResults, num_pixels, pPixels, pDupOf);
                                }
                                else
                                // ==========================================
                                // ORIGINAL CODE (Non-Mode 6)
                                // ==========================================
                                {
                                    pbits[0] = 0; pbits[1] = 0;
                                    evaluate_solution(&lo[0], &hi[0], pbits, pParams, pResults, num_pixels, pPixels, pDupOf);

                                    pbits[0] = 1; pbits[1] = 1;
                                    evaluate_solution(&lo[1], &hi[1], pbits, pParams, pResults, num_pixels, pPixels, pDupOf);
                                }
                        }
                }
                else
                {
                        // compensated rounding
                        const  int iscalep = (1 << (pParams->m_comp_bits + 1)) - 1;
                        const  float scalep = (float)iscalep;

                        const  int32_t totalComps = pParams->m_has_alpha ? 4 : 3;

                        uint32_t best_pbits[2];
                        color_quad_i bestMinColor, bestMaxColor;
                                                
                        if (!pParams->m_endpoints_share_pbit)
                        {
                                float best_err0 = 1e+9;
                                float best_err1 = 1e+9;
                                                                
                                for ( int p = 0; p < 2; p++)
                                {
                                        color_quad_i xMinColor, xMaxColor;

                                        // Notes: The pbit controls which quantization intervals are selected.
                                        // total_levels=2^(comp_bits+1), where comp_bits=4 for mode 0, etc.
                                        // pbit 0: v=(b*2)/(total_levels-1), pbit 1: v=(b*2+1)/(total_levels-1) where b is the component bin from [0,total_levels/2-1] and v is the [0,1] component value
                                        // rearranging you get for pbit 0: b=floor(v*(total_levels-1)/2+.5)
                                        // rearranging you get for pbit 1: b=floor((v*(total_levels-1)-1)/2+.5)
                                        for ( uint32_t c = 0; c < 4; c++)
                                        {
                                                xMinColor.m_c[c] = (int)((xl.m_c[c] * scalep - p) / 2.0f + .5f) * 2 + p;
                                                xMinColor.m_c[c] = ispc_clamp(xMinColor.m_c[c], p, iscalep - 1 + p);

                                                xMaxColor.m_c[c] = (int)((xh.m_c[c] * scalep - p) / 2.0f + .5f) * 2 + p;
                                                xMaxColor.m_c[c] = ispc_clamp(xMaxColor.m_c[c], p, iscalep - 1 + p);
                                        }
                                                                                                                                                                
                                        color_quad_i scaledLow = scale_color(&xMinColor, pParams);
                                        color_quad_i scaledHigh = scale_color(&xMaxColor, pParams);

                                        float err0 = 0;
                                        float err1 = 0;
                                        for ( int i = 0; i < totalComps; i++)
                                        {
                                                err0 += square(scaledLow.m_c[i] - xl.m_c[i]*255.0f);
                                                err1 += square(scaledHigh.m_c[i] - xh.m_c[i]*255.0f);
                                        }

                                        if (err0 < best_err0)
                                        {
                                                best_err0 = err0;
                                                best_pbits[0] = p;
                                                
                                                bestMinColor.m_c[0] = xMinColor.m_c[0] >> 1;
                                                bestMinColor.m_c[1] = xMinColor.m_c[1] >> 1;
                                                bestMinColor.m_c[2] = xMinColor.m_c[2] >> 1;
                                                bestMinColor.m_c[3] = xMinColor.m_c[3] >> 1;
                                        }

                                        if (err1 < best_err1)
                                        {
                                                best_err1 = err1;
                                                best_pbits[1] = p;

                                                bestMaxColor.m_c[0] = xMaxColor.m_c[0] >> 1;
                                                bestMaxColor.m_c[1] = xMaxColor.m_c[1] >> 1;
                                                bestMaxColor.m_c[2] = xMaxColor.m_c[2] >> 1;
                                                bestMaxColor.m_c[3] = xMaxColor.m_c[3] >> 1;
                                        }
                                }
                        }
                        else
                        {
                                // Endpoints share pbits
                                float best_err = 1e+9;

                                for ( int p = 0; p < 2; p++)
                                {
                                        color_quad_i xMinColor, xMaxColor;
                                                                
                                        for ( uint32_t c = 0; c < 4; c++)
                                        {
                                                xMinColor.m_c[c] = (int)((xl.m_c[c] * scalep - p) / 2.0f + .5f) * 2 + p;
                                                xMinColor.m_c[c] = ispc_clamp(xMinColor.m_c[c], p, iscalep - 1 + p);

                                                xMaxColor.m_c[c] = (int)((xh.m_c[c] * scalep - p) / 2.0f + .5f) * 2 + p;
                                                xMaxColor.m_c[c] = ispc_clamp(xMaxColor.m_c[c], p, iscalep - 1 + p);
                                        }
                                                                                
                                        color_quad_i scaledLow = scale_color(&xMinColor, pParams);
                                        color_quad_i scaledHigh = scale_color(&xMaxColor, pParams);

                                        float err = 0;
                                        for ( int i = 0; i < totalComps; i++)
                                                err += square((scaledLow.m_c[i]/255.0f) - xl.m_c[i]) + square((scaledHigh.m_c[i]/255.0f) - xh.m_c[i]);

                                        if (err < best_err)
                                        {
                                                best_err = err;
                                                best_pbits[0] = p;
                                                best_pbits[1] = p;
                                                
                                                bestMinColor.m_c[0] = xMinColor.m_c[0] >> 1;
                                                bestMinColor.m_c[1] = xMinColor.m_c[1] >> 1;
                                                bestMinColor.m_c[2] = xMinColor.m_c[2] >> 1;
                                                bestMinColor.m_c[3] = xMinColor.m_c[3] >> 1;

                                                bestMaxColor.m_c[0] = xMaxColor.m_c[0] >> 1;
                                                bestMaxColor.m_c[1] = xMaxColor.m_c[1] >> 1;
                                                bestMaxColor.m_c[2] = xMaxColor.m_c[2] >> 1;
                                                bestMaxColor.m_c[3] = xMaxColor.m_c[3] >> 1;
                                        }
                                }
                        }

                        fixDegenerateEndpoints(mode, &bestMinColor, &bestMaxColor, &xl, &xh, iscalep >> 1);

                        if ((pResults->m_best_overall_err == UINT64_MAX) || color_quad_i_notequals(&bestMinColor, &pResults->m_low_endpoint) || color_quad_i_notequals(&bestMaxColor, &pResults->m_high_endpoint) || (best_pbits[0] != pResults->m_pbits[0]) || (best_pbits[1] != pResults->m_pbits[1]))
                        {
                                evaluate_solution(&bestMinColor, &bestMaxColor, best_pbits, pParams, pResults, num_pixels, pPixels, pDupOf);
                        }
                }
        }
        else
        {
                const  int iscale = (1 << pParams->m_comp_bits) - 1;
                const  float scale = (float)iscale;

                color_quad_i trialMinColor, trialMaxColor;
                color_quad_i_set_clamped(&trialMinColor, (int)(xl.m_c[0] * scale + .5f), (int)(xl.m_c[1] * scale + .5f), (int)(xl.m_c[2] * scale + .5f), (int)(xl.m_c[3] * scale + .5f));
                color_quad_i_set_clamped(&trialMaxColor, (int)(xh.m_c[0] * scale + .5f), (int)(xh.m_c[1] * scale + .5f), (int)(xh.m_c[2] * scale + .5f), (int)(xh.m_c[3] * scale + .5f));

                fixDegenerateEndpoints(mode, &trialMinColor, &trialMaxColor, &xl, &xh, iscale);

                if ((pResults->m_best_overall_err == UINT64_MAX) || color_quad_i_notequals(&trialMinColor, &pResults->m_low_endpoint) || color_quad_i_notequals(&trialMaxColor, &pResults->m_high_endpoint))
                {
                        uint32_t pbits[2];
                        pbits[0] = 0;
                        pbits[1] = 0;

                        evaluate_solution(&trialMinColor, &trialMaxColor, pbits, pParams, pResults, num_pixels, pPixels, pDupOf);
                }
        }

        return pResults->m_best_overall_err;
}

// Note: In mode 6, m_has_alpha will only be true for transparent blocks.
static uint64_t color_cell_compression( uint32_t mode, const  color_cell_compressor_params * pParams,  color_cell_compressor_results * pResults, 
                const  bc7e_compress_block_params * pComp_params, uint32_t num_pixels, const  color_quad_i * pPixels,  bool refinement)
{
                pResults->m_best_overall_err = UINT64_MAX;

                if ((mode != 6) && (mode != 7))
                {
                                assert(!pParams->m_has_alpha);
                }

                if ((mode <= 2) || (mode == 4) || (mode >= 6))
                {
                                const uint32_t cr = pPixels[0].m_c[0];
                                const uint32_t cg = pPixels[0].m_c[1];
                                const uint32_t cb = pPixels[0].m_c[2];
                                const uint32_t ca = pPixels[0].m_c[3];

                                bool allSame = true;
                                for ( uint32_t i = 1; i < num_pixels; i++)
                                {
                                                if ((cr != pPixels[i].m_c[0]) || (cg != pPixels[i].m_c[1]) || (cb != pPixels[i].m_c[2]) || (ca != pPixels[i].m_c[3]))
                                                {
                                                                allSame = false;
                                                                break;
                                                }
                                }

                                if (allSame)
                                {
                                                if (mode == 0)
                                                                return pack_mode0_to_one_color(pParams, pResults, cr, cg, cb, pResults->m_pSelectors, num_pixels, pPixels);
                                                if (mode == 1)
                                                                return pack_mode1_to_one_color(pParams, pResults, cr, cg, cb, pResults->m_pSelectors, num_pixels, pPixels);
                                                else if (mode == 6)
                                                                return pack_mode6_to_one_color(pParams, pResults, cr, cg, cb, ca, pResults->m_pSelectors, num_pixels, pPixels);
                                                else if (mode == 7)
                                                                return pack_mode7_to_one_color(pParams, pResults, cr, cg, cb, ca, pResults->m_pSelectors, num_pixels, pPixels);
                                                else
                                                                return pack_mode24_to_one_color(pParams, pResults, cr, cg, cb, pResults->m_pSelectors, num_pixels, pPixels);
                                }
                }


                // PIXEL DEDUPLICATION CACHE (hoisted from evaluate_solution)
                // Computed once per subset, reused across all find_optimal_solution calls.
                int dup_of_cc[16];
                {
                        const uint32_t nc_dup = pParams->m_has_alpha ? 4 : 3;
                        for (uint32_t i = 0; i < num_pixels; i++) {
                                dup_of_cc[i] = -1;
                                for (uint32_t j = 0; j < i; j++) {
                                        bool same = true;
                                        for (uint32_t c = 0; c < nc_dup; c++) {
                                                if (pPixels[i].m_c[c] != pPixels[j].m_c[c]) { same = false; break; }
                                        }
                                        if (same) { dup_of_cc[i] = (int)j; break; }
                                }
                        }
                }
                vec4F meanColor, axis;
                vec4F_set_scalar(&meanColor, 0.0f);

                for ( uint32_t i = 0; i < num_pixels; i++)
                {
                                vec4F color = vec4F_from_color(&pPixels[i]);
                                meanColor = vec4F_add(&meanColor, &color);
                }
                                                                
                vec4F meanColorScaled = vec4F_mul(&meanColor, 1.0f / (float)((int)num_pixels));

                meanColor = vec4F_mul(&meanColor, 1.0f / (float)((int)num_pixels * 255.0f));
                vec4F_saturate_in_place(&meanColor);

                if (pParams->m_has_alpha)
                {
                                vec4F v;
                                vec4F_set_scalar(&v, 0.0f);
                                for ( uint32_t i = 0; i < num_pixels; i++)
                                {
                                                vec4F color = vec4F_from_color(&pPixels[i]);
                                                color = vec4F_sub(&color, &meanColorScaled);

                                                vec4F a = vec4F_mul(&color, color.m_c[0]);
                                                vec4F b = vec4F_mul(&color, color.m_c[1]);
                                                vec4F c = vec4F_mul(&color, color.m_c[2]);
                                                vec4F d = vec4F_mul(&color, color.m_c[3]);

                                                vec4F n = i ? v : color;
                                                vec4F_normalize_in_place(&n);

                                                v.m_c[0] += vec4F_dot(&a, &n);
                                                v.m_c[1] += vec4F_dot(&b, &n);
                                                v.m_c[2] += vec4F_dot(&c, &n);
                                                v.m_c[3] += vec4F_dot(&d, &n);
                                }
                                axis = v;
                                vec4F_normalize_in_place(&axis);
                }
                else
                {
                                float cov[6];
                                cov[0] = 0; cov[1] = 0; cov[2] = 0; 
                                cov[3] = 0; cov[4] = 0; cov[5] = 0;

                                for ( uint32_t i = 0; i < num_pixels; i++)
                                {
                                                const  color_quad_i * pV = &pPixels[i];

                                                float r = pV->m_c[0] - meanColorScaled.m_c[0];
                                                float g = pV->m_c[1] - meanColorScaled.m_c[1];
                                                float b = pV->m_c[2] - meanColorScaled.m_c[2];
                                                                
                                                cov[0] += r*r;
                                                cov[1] += r*g;
                                                cov[2] += r*b;
                                                cov[3] += g*g;
                                                cov[4] += g*b;
                                                cov[5] += b*b;
                                }

                                float vfr, vfg, vfb;
                                //vfr = hi[0] - lo[0];
                                //vfg = hi[1] - lo[1];
                                //vfb = hi[2] - lo[2];
                                // This is more stable.
                                vfr = .9f;
                                vfg = 1.0f;
                                vfb = .7f;

                                for ( uint32_t iter = 0; iter < 1; iter++)
                                {
                                                float r = vfr*cov[0] + vfg*cov[1] + vfb*cov[2];
                                                float g = vfr*cov[1] + vfg*cov[3] + vfb*cov[4];
                                                float b = vfr*cov[2] + vfg*cov[4] + vfb*cov[5];

                                                float m = maximumf(maximumf(abs(r), abs(g)), abs(b));
                                                if (m > 1e-10f)
                                                {
                                                                m = 1.0f / m;
                                                                r *= m;
                                                                g *= m;
                                                                b *= m;
                                                }

                                                //float delta = square(vfr - r) + square(vfg - g) + square(vfb - b);

                                                vfr = r;
                                                vfg = g;
                                                vfb = b;

                                                //if ((iter > 1) && (delta < 1e-8f))
                                                //      break;
                                }

                                float len = vfr*vfr + vfg*vfg + vfb*vfb;

                                if (len < 1e-10f)
                                                vec4F_set_scalar(&axis, 0.0f);
                                else
                                {
                                                len = 1.0f / sqrt(len);
                                                vfr *= len;
                                                vfg *= len;
                                                vfb *= len;
                                                vec4F_set(&axis, vfr, vfg, vfb, 0);
                                }
                }

                if (vec4F_dot(&axis, &axis) < .5f)
                {
                                if (pParams->m_perceptual)
                                                vec4F_set(&axis, .213f, .715f, .072f, pParams->m_has_alpha ? .715f : 0);
                                else
                                                vec4F_set(&axis, 1.0f, 1.0f, 1.0f, pParams->m_has_alpha ? 1.0f : 0);
                                vec4F_normalize_in_place(&axis);
                }

                float l = 1e+9f, h = -1e+9f;

                for ( uint32_t i = 0; i < num_pixels; i++)
                {
                                vec4F color = vec4F_from_color(&pPixels[i]);

                                vec4F q = vec4F_sub(&color, &meanColorScaled);
                                float d = vec4F_dot(&q, &axis);

                                l = minimumf(l, d);
                                h = maximumf(h, d);
                }

                l *= (1.0f / 255.0f);
                h *= (1.0f / 255.0f);

                vec4F b0 = vec4F_mul(&axis, l);
                vec4F b1 = vec4F_mul(&axis, h);
                vec4F c0 = vec4F_add(&meanColor, &b0);
                vec4F c1 = vec4F_add(&meanColor, &b1);
                vec4F minColor = vec4F_saturate(&c0);
                vec4F maxColor = vec4F_saturate(&c1);
                                                                
                vec4F whiteVec;
                vec4F_set_scalar(&whiteVec, 1.0f);
                if (vec4F_dot(&minColor, &whiteVec) > vec4F_dot(&maxColor, &whiteVec))
                {
                                vec4F temp = minColor;
                                minColor = maxColor;
                                maxColor = temp;
                }

                if (!find_optimal_solution(mode, &minColor, &maxColor, pParams, pResults, pComp_params->m_pbit_search, num_pixels, pPixels, dup_of_cc))
                                return 0;

                // SKIP REFINEMENT IF ALREADY PERFECT OR NEAR-PERFECT
                // If the initial PCA nailed it, don't waste time on refinement.
                if (!refinement || pResults->m_best_overall_err == 0)
                        return pResults->m_best_overall_err;

                // For very low error blocks, skip expensive uber refinement
                // (saves the 7 extra find_optimal_solution calls in uber_level > 0)
                // Threshold: ~0.5 RMSE per pixel is visually lossless
                const uint64_t cheap_refine_thresh = (uint32_t)((uint32_t)num_pixels * 1);
                if (pResults->m_best_overall_err <= cheap_refine_thresh)
                        return pResults->m_best_overall_err;
                
                // OPT-A (speed): Refinement pass early-exit. Each refinement pass
                // computes least-squares endpoints from the current best selectors,
                // then re-runs find_optimal_solution. If a pass doesn't improve
                // pResults->m_best_overall_err, the selectors are unchanged, so
                // the next pass's least-squares will produce the same endpoints
                // and find_optimal_solution will give the same result. Breaking
                // out is provably output-identical to running all passes.
                uint64_t prev_refine_err = pResults->m_best_overall_err;
                for ( uint32_t i = 0; i < pComp_params->m_refinement_passes; i++)
                {
                                vec4F xl, xh;
                                vec4F_set_scalar(&xl, 0.0f);
                                vec4F_set_scalar(&xh, 0.0f);
                                if (pParams->m_has_alpha)
                                                compute_least_squares_endpoints_rgba(num_pixels, pResults->m_pSelectors, pParams->m_pSelector_weightsx, &xl, &xh, pPixels);
                                else
                                {
                                                compute_least_squares_endpoints_rgb(num_pixels, pResults->m_pSelectors, pParams->m_pSelector_weightsx, &xl, &xh, pPixels);

                                                xl.m_c[3] = 255.0f;
                                                xh.m_c[3] = 255.0f;
                                }

                                xl = vec4F_mul(&xl, (1.0f / 255.0f));
                                xh = vec4F_mul(&xh, (1.0f / 255.0f));

                                if (!find_optimal_solution(mode, &xl, &xh, pParams, pResults, pComp_params->m_pbit_search, num_pixels, pPixels, dup_of_cc))
                                                return 0;

                                // OPT-A: stop refining once convergence stalls
                                if (pResults->m_best_overall_err >= prev_refine_err)
                                                break;
                                prev_refine_err = pResults->m_best_overall_err;
                }

                if (pComp_params->m_uber_level > 0)
                {
                                int selectors_temp[16], selectors_temp1[16];
                                for ( uint32_t i = 0; i < num_pixels; i++)
                                                selectors_temp[i] = pResults->m_pSelectors[i];

                                const  int max_selector = pParams->m_num_selector_weights - 1;

                                uint32_t min_sel = 16;
                                uint32_t max_sel = 0;
                                for ( uint32_t i = 0; i < num_pixels; i++)
                                {
                                                uint32_t sel = selectors_temp[i];
                                                min_sel = minimumu(min_sel, sel);
                                                max_sel = maximumu(max_sel, sel);
                                }

                                vec4F xl, xh;
                                vec4F_set_scalar(&xl, 0.0f);
                                vec4F_set_scalar(&xh, 0.0f);

                                if (pComp_params->m_uber1_mask & 1)
                                {
                                                for ( uint32_t i = 0; i < num_pixels; i++)
                                                {
                                                                uint32_t sel = selectors_temp[i];
                                                                if ((sel == min_sel) && (sel < (pParams->m_num_selector_weights - 1)))
                                                                                sel++;
                                                                selectors_temp1[i] = sel;
                                                }
                                                                                                
                                                if (pParams->m_has_alpha)
                                                                compute_least_squares_endpoints_rgba(num_pixels, selectors_temp1, pParams->m_pSelector_weightsx, &xl, &xh, pPixels);
                                                else
                                                {
                                                                compute_least_squares_endpoints_rgb(num_pixels, selectors_temp1, pParams->m_pSelector_weightsx, &xl, &xh, pPixels);
                                                                xl.m_c[3] = 255.0f;
                                                                xh.m_c[3] = 255.0f;
                                                }

                                                xl = vec4F_mul(&xl, (1.0f / 255.0f));
                                                xh = vec4F_mul(&xh, (1.0f / 255.0f));

                                                if (!find_optimal_solution(mode, &xl, &xh, pParams, pResults, pComp_params->m_pbit_search, num_pixels, pPixels, dup_of_cc))
                                                                return 0;
                                }

                                if (pComp_params->m_uber1_mask & 2)
                                {
                                                for ( uint32_t i = 0; i < num_pixels; i++)
                                                {
                                                                uint32_t sel = selectors_temp[i];
                                                                if ((sel == max_sel) && (sel > 0))
                                                                                sel--;
                                                                selectors_temp1[i] = sel;
                                                }

                                                if (pParams->m_has_alpha)
                                                                compute_least_squares_endpoints_rgba(num_pixels, selectors_temp1, pParams->m_pSelector_weightsx, &xl, &xh, pPixels);
                                                else
                                                {
                                                                compute_least_squares_endpoints_rgb(num_pixels, selectors_temp1, pParams->m_pSelector_weightsx, &xl, &xh, pPixels);
                                                                xl.m_c[3] = 255.0f;
                                                                xh.m_c[3] = 255.0f;
                                                }

                                                xl = vec4F_mul(&xl, (1.0f / 255.0f));
                                                xh = vec4F_mul(&xh, (1.0f / 255.0f));

                                                if (!find_optimal_solution(mode, &xl, &xh, pParams, pResults, pComp_params->m_pbit_search, num_pixels, pPixels, dup_of_cc))
                                                                return 0;
                                }

                                if (pComp_params->m_uber1_mask & 4)
                                {
                                                for ( uint32_t i = 0; i < num_pixels; i++)
                                                {
                                                                uint32_t sel = selectors_temp[i];
                                                                if ((sel == min_sel) && (sel < (pParams->m_num_selector_weights - 1)))
                                                                                sel++;
                                                                else if ((sel == max_sel) && (sel > 0))
                                                                                sel--;
                                                                selectors_temp1[i] = sel;
                                                }

                                                if (pParams->m_has_alpha)
                                                                compute_least_squares_endpoints_rgba(num_pixels, selectors_temp1, pParams->m_pSelector_weightsx, &xl, &xh, pPixels);
                                                else
                                                {
                                                                compute_least_squares_endpoints_rgb(num_pixels, selectors_temp1, pParams->m_pSelector_weightsx, &xl, &xh, pPixels);
                                                                xl.m_c[3] = 255.0f;
                                                                xh.m_c[3] = 255.0f;
                                                }

                                                xl = vec4F_mul(&xl, (1.0f / 255.0f));
                                                xh = vec4F_mul(&xh, (1.0f / 255.0f));

                                                if (!find_optimal_solution(mode, &xl, &xh, pParams, pResults, pComp_params->m_pbit_search, num_pixels, pPixels, dup_of_cc))
                                                                return 0;
                                }

                                const uint32_t uber_err_thresh = (num_pixels * 56) >> 4;
                                if ((pComp_params->m_uber_level >= 2) && (pResults->m_best_overall_err > uber_err_thresh))
                                {
                                                const  int Q = (pComp_params->m_uber_level >= 4) ? (pComp_params->m_uber_level - 2) : 1;
                                                for ( int ly = -Q; ly <= 1; ly++)
                                                {
                                                                for ( int hy = max_selector - 1; hy <= (max_selector + Q); hy++)
                                                                {
                                                                                if ((ly == 0) && (hy == max_selector))
                                                                                                continue;

                                                                                for ( uint32_t i = 0; i < num_pixels; i++)
                                                                                                selectors_temp1[i] = (int)clampf(floor((float)max_selector * ((float)(int)selectors_temp[i] - (float)ly) / ((float)hy - (float)ly) + .5f), 0, (float)max_selector);

                                                                                vec4F_set_scalar(&xl, 0.0f);
                                                                                vec4F_set_scalar(&xh, 0.0f);
                                                                                if (pParams->m_has_alpha)
                                                                                                compute_least_squares_endpoints_rgba(num_pixels, selectors_temp1, pParams->m_pSelector_weightsx, &xl, &xh, pPixels);
                                                                                else
                                                                                {
                                                                                                compute_least_squares_endpoints_rgb(num_pixels, selectors_temp1, pParams->m_pSelector_weightsx, &xl, &xh, pPixels);
                                                                                                xl.m_c[3] = 255.0f;
                                                                                                xh.m_c[3] = 255.0f;
                                                                                }

                                                                                xl = vec4F_mul(&xl, (1.0f / 255.0f));
                                                                                xh = vec4F_mul(&xh, (1.0f / 255.0f));

                                                                                if (!find_optimal_solution(mode, &xl, &xh, pParams, pResults, pComp_params->m_pbit_search && (pComp_params->m_uber_level >= 2), num_pixels, pPixels, dup_of_cc))
                                                                                                return 0;
                                                                }
                                                }
                                }
                }

                // GREEN DC BIAS CORRECTION (Fix GC3): per-block targeted endpoint
                // nudge for low-precision modes (Mode 4/7, comp_bits=5, 6-bit grid
                // with pbit, step 4 in 8-bit space).
                //
                // PROBLEM: Mode 4/7 endpoints snap to a 6-bit grid (step 4 in 8-bit).
                // For blocks where the true endpoint falls near a grid midpoint (e.g.
                // true=30, grid points 28 and 32), BOTH grid points give the same
                // squared error. The encoder's round-half-up picks the HIGHER grid
                // point (32), giving +2 bias. When BOTH endpoints have this bias,
                // the block gets uniform +2 to +4 green DC bias — visible as
                // "brightening patches."
                //
                // FIX: After all refinement, compute the block's green DC bias. If
                // |bias| > 1.5, try shifting BOTH green endpoints by 1 grid step
                // (4 in 8-bit) in the direction OPPOSITE to bias. This is done by
                // manipulating the 6-bit (endpoint<<1 | pbit) value:
                //   - Shift DOWN: if pbit=1, flip to 0; else endpoint-=1, pbit=1
                //   - Shift UP: if pbit=0, flip to 1; else endpoint+=1, pbit=0
                // For shared-pbit modes, both endpoints use the same pbit, so the
                // shift moves both by the same amount.
                //
                // We accept the shift if the new error is <= the old error (tie-
                // breaking) OR within a small tolerance (1.25x). This catches the
                // common case where both grid points give equal error but different
                // bias, plus the case where a small error increase yields a large
                // bias reduction.
                //
                // PERF: 1 extra evaluate_solution call per affected block (~1.5%).
                // Total overhead < 0.1%.
                // Covers Mode 0/2/4/7 (comp_bits<=5 with pbit = 6-bit grid, step 4).
                // Mode 1 (comp_bits=6+pbit = 7-bit, step 2) and Mode 3/5 (comp_bits=7,
                // step 2) have at most ±1 endpoint bias — below the 1.5 threshold.
                // Mode 1's high bias comes from SELECTOR errors (not endpoints), which
                // GC3 cannot fix.
                if (pResults->m_best_overall_err != UINT64_MAX &&
                    pParams->m_comp_bits <= 5 &&
                    pParams->m_has_pbits &&
                    !pParams->m_perceptual)
                {
                        // Decode current green endpoints to 6-bit (full value including pbit)
                        color_quad_i qLow = pResults->m_low_endpoint;
                        color_quad_i qHigh = pResults->m_high_endpoint;
                        uint32_t cur_minPBit = 0, cur_maxPBit = 0;
                        uint32_t low_6bit, high_6bit;

                        if (pParams->m_has_pbits)
                        {
                                if (pParams->m_endpoints_share_pbit) cur_maxPBit = cur_minPBit = pResults->m_pbits[0];
                                else { cur_minPBit = pResults->m_pbits[0]; cur_maxPBit = pResults->m_pbits[1]; }
                                low_6bit = (qLow.m_c[1] << 1) | cur_minPBit;
                                high_6bit = (qHigh.m_c[1] << 1) | cur_maxPBit;
                        }
                        else
                        {
                                // No pbit: endpoint IS the full n-bit value (6-bit for Mode 1)
                                low_6bit = qLow.m_c[1];
                                high_6bit = qHigh.m_c[1];
                        }

                        // Scale to 8-bit: n=6 for both comp_bits=5+pbit and comp_bits=6-no-pbit
                        // scale_color formula: v = c << (8-n); v |= v >> n. For n=6: v = c<<2 | c>>4
                        uint32_t aLowG = (low_6bit << 2) | (low_6bit >> 4);
                        uint32_t aHighG = (high_6bit << 2) | (high_6bit >> 4);

                        // Compute decoded green DC and input green DC
                        int decoded_g_sum = 0, input_g_sum = 0;
                        for (uint32_t i = 0; i < num_pixels; i++)
                        {
                                uint32_t sel = pResults->m_pSelectors[i];
                                uint32_t w = pParams->m_pSelector_weights[sel];
                                int dg = (int)(((64 - w) * aLowG + w * aHighG + 32) >> 6);
                                decoded_g_sum += dg;
                                input_g_sum += pPixels[i].m_c[1];
                        }
                        float bias = (float)(decoded_g_sum - input_g_sum) / (float)num_pixels;

                        if (bias > 1.5f || bias < -1.5f)
                        {
                                int shift_dir = (bias > 0) ? -1 : 1;
                                const int max_6bit = 63;  // 6-bit max

                                int new_low_6bit = (int)low_6bit + shift_dir;
                                int new_high_6bit = (int)high_6bit + shift_dir;

                                // Clamp check
                                if (new_low_6bit >= 0 && new_low_6bit <= max_6bit &&
                                        new_high_6bit >= 0 && new_high_6bit <= max_6bit)
                                {
                                        // Decompose back to (endpoint, pbit)
                                        uint32_t new_low_ep, new_low_pbit, new_high_ep, new_high_pbit;
                                        bool valid = true;

                                        if (pParams->m_has_pbits)
                                        {
                                                new_low_pbit = new_low_6bit & 1;
                                                new_high_pbit = new_high_6bit & 1;
                                                new_low_ep = new_low_6bit >> 1;
                                                new_high_ep = new_high_6bit >> 1;
                                                // For shared pbit, both must match
                                                if (pParams->m_endpoints_share_pbit && new_low_pbit != new_high_pbit)
                                                        valid = false;
                                        }
                                        else
                                        {
                                                new_low_ep = new_low_6bit;
                                                new_high_ep = new_high_6bit;
                                                new_low_pbit = 0;
                                                new_high_pbit = 0;
                                        }

                                        if (valid)
                                        {
                                                color_quad_i new_low = pResults->m_low_endpoint;
                                                color_quad_i new_high = pResults->m_high_endpoint;
                                                new_low.m_c[1] = new_low_ep;
                                                new_high.m_c[1] = new_high_ep;
                                                uint32_t new_pbits[2] = { new_low_pbit, new_high_pbit };

                                                uint64_t saved_err = pResults->m_best_overall_err;
                                                color_quad_i saved_low = pResults->m_low_endpoint;
                                                color_quad_i saved_high = pResults->m_high_endpoint;
                                                uint32_t saved_pbits[2] = { pResults->m_pbits[0], pResults->m_pbits[1] };
                                                int saved_selectors[16];
                                                for (uint32_t i = 0; i < num_pixels; i++) saved_selectors[i] = pResults->m_pSelectors[i];

                                                pResults->m_best_overall_err = UINT64_MAX;
                                                evaluate_solution(&new_low, &new_high, new_pbits, pParams, pResults, num_pixels, pPixels, dup_of_cc);

                                                uint64_t new_err = pResults->m_best_overall_err;

                                                // Bias-aware acceptance: compute new bias and accept if
                                                // bias reduction justifies error increase.
                                                uint32_t naLowG = (new_low_6bit << 2) | (new_low_6bit >> 4);
                                                uint32_t naHighG = (new_high_6bit << 2) | (new_high_6bit >> 4);
                                                int new_decoded_g_sum = 0;
                                                for (uint32_t i = 0; i < num_pixels; i++)
                                                {
                                                        uint32_t sel = pResults->m_pSelectors[i];
                                                        uint32_t w = pParams->m_pSelector_weights[sel];
                                                        int dg = (int)(((64 - w) * naLowG + w * naHighG + 32) >> 6);
                                                        new_decoded_g_sum += dg;
                                                }
                                                float new_bias = (float)(new_decoded_g_sum - input_g_sum) / (float)num_pixels;
                                                float bias_reduction = (float)fabs(bias) - (float)fabs(new_bias);
                                                float bias_benefit = bias_reduction * bias_reduction * (float)num_pixels * 2.0f;
                                                int64_t err_increase = (int64_t)new_err - (int64_t)saved_err;
                                                bool accept = (err_increase <= 0) || (bias_benefit > (float)err_increase);

                                                if (!accept)
                                                {
                                                        pResults->m_best_overall_err = saved_err;
                                                        pResults->m_low_endpoint = saved_low;
                                                        pResults->m_high_endpoint = saved_high;
                                                        pResults->m_pbits[0] = saved_pbits[0];
                                                        pResults->m_pbits[1] = saved_pbits[1];
                                                        for (uint32_t i = 0; i < num_pixels; i++) pResults->m_pSelectors[i] = saved_selectors[i];
                                                }
                                        }
                                }
                        }
                }

                return pResults->m_best_overall_err;
}

static uint64_t color_cell_compression_est( uint32_t mode, const  color_cell_compressor_params * pParams, uint64_t best_err_so_far,  uint32_t num_pixels, const  color_quad_i * pPixels)
{
                assert((pParams->m_num_selector_weights == 4) || (pParams->m_num_selector_weights == 8));

                float lr = 255, lg = 255, lb = 255;
                float hr = 0, hg = 0, hb = 0;
                for ( uint32_t i = 0; i < num_pixels; i++)
                {
                                const  color_quad_i * pC = &pPixels[i];

                                float r = pC->m_c[0];
                                float g = pC->m_c[1];
                                float b = pC->m_c[2];
                                
                                lr = min(lr, r);
                                lg = min(lg, g);
                                lb = min(lb, b);

                                hr = max(hr, r);
                                hg = max(hg, g);
                                hb = max(hb, b);
                }
                                                
                const  uint32_t N = 1 << g_bc7_color_index_bitcount[mode];
                                                                                                
                uint64_t total_err = 0;
                
                float sr = lr;
                float sg = lg;
                float sb = lb;

                float dir = hr - lr;
                float dig = hg - lg;
                float dib = hb - lb;    

                float far = dir;
                float fag = dig;
                float fab = dib;

                float low = far * sr + fag * sg + fab * sb;
                float high = far * hr + fag * hg + fab * hb;

                float scale = (high != low) ? (((float)N - 1) / (float)(high - low)) : 0.0f;
                float inv_n = 1.0f / ((float)N - 1);

                float total_errf = 0;

                // We don't handle perceptual very well here, but the difference is very slight (<.05 dB avg Luma PSNR across a large corpus) and the perf lost was high (2x slower).
                if ((pParams->m_weights[0] != 1) || (pParams->m_weights[1] != 1) || (pParams->m_weights[2] != 1))
                {
                                float wr = pParams->m_weights[0];
                                float wg = pParams->m_weights[1];
                                float wb = pParams->m_weights[2];

                                for ( uint32_t i = 0; i < num_pixels; i++)
                                {
                                                const  color_quad_i * pC = &pPixels[i];

                                                float d = far * (float)pC->m_c[0] + fag * (float)pC->m_c[1] + fab * (float)pC->m_c[2];

                                                float s = ispc_clamp(floor((d - low) * scale + .5f) * inv_n, 0.0f, 1.0f);

                                                float itr = sr + dir * s;
                                                float itg = sg + dig * s;
                                                float itb = sb + dib * s;

                                                float dr = itr - (float)pC->m_c[0];
                                                float dg = itg - (float)pC->m_c[1];
                                                float db = itb - (float)pC->m_c[2];

                                                total_errf += wr * dr * dr + wg * dg * dg + wb * db * db;

                                                // ADD THIS:
                                                if ((uint64_t)total_errf >= best_err_so_far)
                                                        return (uint64_t)total_errf;
                                }
                }
                else
                {
                                for ( uint32_t i = 0; i < num_pixels; i++)
                                {
                                                const  color_quad_i * pC = &pPixels[i];

                                                float d = far * (float)pC->m_c[0] + fag * (float)pC->m_c[1] + fab * (float)pC->m_c[2];

                                                float s = ispc_clamp(floor((d - low) * scale + .5f) * inv_n, 0.0f, 1.0f);

                                                float itr = sr + dir * s;
                                                float itg = sg + dig * s;
                                                float itb = sb + dib * s;

                                                float dr = itr - (float)pC->m_c[0];
                                                float dg = itg - (float)pC->m_c[1];
                                                float db = itb - (float)pC->m_c[2];

                                                // NON-PERCEPTUAL LOOP:
                                                total_errf += dr * dr + dg * dg + db * db;
                                                
                                                // PERCEPTUAL LOOP:
                                                // total_errf += wr * dr * dr + wg * dg * dg + wb * db * db;

                                                // =====================================================================
                                                // SPEEDUP: Abort early if this partition is already worse than our best.
                                                // =====================================================================
                                                if ((uint64_t)total_errf >= best_err_so_far) 
                                                                return (uint64_t)total_errf; 
                                }
                }

                total_err = (int64_t)total_errf;

                return total_err;
}

static uint64_t color_cell_compression_est_mode7( uint32_t mode, const  color_cell_compressor_params * pParams, uint64_t best_err_so_far,  uint32_t num_pixels, const  color_quad_i * pPixels)
{
                assert((mode == 7) && (pParams->m_num_selector_weights == 4));

                float lr = 255, lg = 255, lb = 255, la = 255;
                float hr = 0, hg = 0, hb = 0, ha = 0;
                for ( uint32_t i = 0; i < num_pixels; i++)
                {
                                const  color_quad_i * pC = &pPixels[i];

                                float r = pC->m_c[0];
                                float g = pC->m_c[1];
                                float b = pC->m_c[2];
                                float a = pC->m_c[3];
                                
                                lr = min(lr, r);
                                lg = min(lg, g);
                                lb = min(lb, b);
                                la = min(la, a);

                                hr = max(hr, r);
                                hg = max(hg, g);
                                hb = max(hb, b);
                                ha = max(ha, a);
                }
                                                
                const  uint32_t N = 4;
                                                                                                
                uint64_t total_err = 0;
                
                float sr = lr;
                float sg = lg;
                float sb = lb;
                float sa = la;

                float dir = hr - lr;
                float dig = hg - lg;
                float dib = hb - lb;    
                float dia = ha - la;    

                float far = dir;
                float fag = dig;
                float fab = dib;
                float faa = dia;

                float low = far * sr + fag * sg + fab * sb + faa * sa;
                float high = far * hr + fag * hg + fab * hb + faa * ha;

                float scale = (high != low) ? (((float)N - 1) / (float)(high - low)) : 0.0f;
                float inv_n = 1.0f / ((float)N - 1);

                float total_errf = 0;

                // We don't handle perceptual very well here, but the difference is very slight (<.05 dB avg Luma PSNR across a large corpus) and the perf lost was high (2x slower).
                if ( (!pParams->m_perceptual) && ((pParams->m_weights[0] != 1) || (pParams->m_weights[1] != 1) || (pParams->m_weights[2] != 1) || (pParams->m_weights[3] != 1)) )
                {
                                float wr = pParams->m_weights[0];
                                float wg = pParams->m_weights[1];
                                float wb = pParams->m_weights[2];
                                float wa = pParams->m_weights[3];

                                for ( uint32_t i = 0; i < num_pixels; i++)
                                {
                                                const  color_quad_i * pC = &pPixels[i];

                                                float d = far * (float)pC->m_c[0] + fag * (float)pC->m_c[1] + fab * (float)pC->m_c[2] + faa * (float)pC->m_c[3];

                                                float s = ispc_clamp(floor((d - low) * scale + .5f) * inv_n, 0.0f, 1.0f);

                                                float itr = sr + dir * s;
                                                float itg = sg + dig * s;
                                                float itb = sb + dib * s;
                                                float ita = sa + dia * s;

                                                float dr = itr - (float)pC->m_c[0];
                                                float dg = itg - (float)pC->m_c[1];
                                                float db = itb - (float)pC->m_c[2];
                                                float da = ita - (float)pC->m_c[3];

                                                total_errf += wr * dr * dr + wg * dg * dg + wb * db * db + wa * da * da;
                                }
                }
                else
                {
                                for ( uint32_t i = 0; i < num_pixels; i++)
                                {
                                                const  color_quad_i * pC = &pPixels[i];

                                                float d = far * (float)pC->m_c[0] + fag * (float)pC->m_c[1] + fab * (float)pC->m_c[2] + faa * (float)pC->m_c[3];

                                                float s = ispc_clamp(floor((d - low) * scale + .5f) * inv_n, 0.0f, 1.0f);

                                                float itr = sr + dir * s;
                                                float itg = sg + dig * s;
                                                float itb = sb + dib * s;
                                                float ita = sa + dia * s;

                                                float dr = itr - (float)pC->m_c[0];
                                                float dg = itg - (float)pC->m_c[1];
                                                float db = itb - (float)pC->m_c[2];
                                                float da = ita - (float)pC->m_c[3];

                                                total_errf += dr * dr + dg * dg + db * db + da * da;
                                }
                }

                total_err = (int64_t)total_errf;

                return total_err;
}

static uint32_t estimate_partition( uint32_t mode, const  color_quad_i * pPixels, const  bc7e_compress_block_params * pComp_params)
{
                const  uint32_t total_subsets = g_bc7_num_subsets[mode];
                 uint32_t total_partitions = minimumu(pComp_params->m_max_partitions_mode[mode], 1U << g_bc7_partition_bits[mode]);

                if (total_partitions <= 1)
                                return 0;

                uint64_t best_err = UINT64_MAX;
                uint32_t best_partition = 0;

                 color_cell_compressor_params params;
                color_cell_compressor_params_clear(&params);

                params.m_pSelector_weights = (g_bc7_color_index_bitcount[mode] == 2) ? g_bc7_weights2 : g_bc7_weights3;
                params.m_num_selector_weights = 1 << g_bc7_color_index_bitcount[mode];

                memcpy(params.m_weights, pComp_params->m_weights, sizeof(params.m_weights));
                
                if (mode >= 6)
                {
                                params.m_weights[0] *= pComp_params->m_alpha_settings.m_mode67_error_weight_mul[0];
                                params.m_weights[1] *= pComp_params->m_alpha_settings.m_mode67_error_weight_mul[1];
                                params.m_weights[2] *= pComp_params->m_alpha_settings.m_mode67_error_weight_mul[2];
                                params.m_weights[3] *= pComp_params->m_alpha_settings.m_mode67_error_weight_mul[3];
                }

                params.m_perceptual = pComp_params->m_perceptual;

                for ( uint32_t partition = 0; partition < total_partitions; partition++)
                {
                                const int * pPartition = (total_subsets == 3) ? &g_bc7_partition3[partition * 16] : &g_bc7_partition2[partition * 16];

                                 color_quad_i subset_colors[3][16];
                                 uint32_t subset_total_colors[3];
                                subset_total_colors[0] = 0;
                                subset_total_colors[1] = 0;
                                subset_total_colors[2] = 0;
                                
                                for ( uint32_t index = 0; index < 16; index++)
                                {
                                                const  uint32_t p = pPartition[index];

                                                subset_colors[p][subset_total_colors[p]] = pPixels[index];
                                                subset_total_colors[p]++;
                                }

                                uint64_t total_subset_err = 0;

                                for ( uint32_t subset = 0; subset < total_subsets; subset++)
                                {
                                                uint64_t err;
                                                if (mode == 7)
                                                                err = color_cell_compression_est_mode7(mode, &params, best_err, subset_total_colors[subset], &subset_colors[subset][0]);
                                                else
                                                                err = color_cell_compression_est(mode, &params, best_err, subset_total_colors[subset], &subset_colors[subset][0]);

                                                total_subset_err += err;

                                } // subset

                                if (total_subset_err < best_err)
                                {
                                                best_err = total_subset_err;
                                                best_partition = partition;
                                                if (!best_err)
                                                                break;
                                }

                                if (total_subsets == 2)
                                {
                                                if ((partition == BC7E_2SUBSET_CHECKERBOARD_PARTITION_INDEX) && (best_partition != BC7E_2SUBSET_CHECKERBOARD_PARTITION_INDEX))
                                                                break;
                                }

                } // partition

                return best_partition;
}

struct solution
{
                uint32_t m_index;
                uint64_t m_err;
};

// Phase 2 candidate
struct phase2_candidate
{
        uint32_t partition;
        uint64_t score;
        int cached_min_r[3], cached_max_r[3];
        int cached_min_g[3], cached_max_g[3];
        int cached_min_b[3], cached_max_b[3];
};

static uint32_t estimate_partition_list(uint32_t mode, const color_quad_i * pPixels,
    const bc7e_compress_block_params * pComp_params,
    solution * pSolutions, int32_t max_solutions,
    bool alpha_aware_volume = false)
{
    const int32_t orig_max_solutions = max_solutions;
    const uint32_t total_subsets = g_bc7_num_subsets[mode];
    uint32_t total_partitions = minimumu(pComp_params->m_max_partitions_mode[mode],
                                          1U << g_bc7_partition_bits[mode]);

    if (total_partitions <= 1)
    {
        pSolutions[0].m_index = 0;
        pSolutions[0].m_err = 0;
        return 1;
    }
    else if (max_solutions >= total_partitions)
    {
        for (int i = 0; i < (int)total_partitions; i++)
        {
            pSolutions[i].m_index = i;
            pSolutions[i].m_err = i;
        }
        return total_partitions;
    }

    const int32_t HIGH_FREQUENCY_SORTED_PARTITION_THRESHOLD = 4;
    if (total_subsets == 2)
    {
        if (max_solutions < HIGH_FREQUENCY_SORTED_PARTITION_THRESHOLD)
            max_solutions = HIGH_FREQUENCY_SORTED_PARTITION_THRESHOLD;
    }

    // ============================================================
    // PASS 1: Cheap volume heuristic pre-filter
    // Reduce 64 partitions to ~8-12 candidates using min/max only.
    // No floating-point, no PCA. ~10x faster than est().
    // ============================================================
    const uint32_t PREFILTER_K = max_solutions * 4 + 4;
    const uint32_t PREFILTER_MAX = (PREFILTER_K > total_partitions) ? total_partitions : PREFILTER_K;

    // Stage 1: compute volume scores for all partitions
    uint64_t vol_scores[64];
    for (uint32_t p = 0; p < total_partitions; p++)
    {
        const int *part = (total_subsets == 3) ?
            &g_bc7_partition3[p * 16] : &g_bc7_partition2[p * 16];

        int min_r[3] = {255, 255, 255}, max_r[3] = {0, 0, 0};
        int min_g[3] = {255, 255, 255}, max_g[3] = {0, 0, 0};
        int min_b[3] = {255, 255, 255}, max_b[3] = {0, 0, 0};
        int min_a[3] = {255, 255, 255}, max_a[3] = {0, 0, 0};
        int counts[3] = {0, 0, 0};

        for (int i = 0; i < 16; i++)
        {
            int s = part[i];
            int r = pPixels[i].m_c[0], g = pPixels[i].m_c[1], b = pPixels[i].m_c[2];
            int a = pPixels[i].m_c[3];
            counts[s]++;
            if (r < min_r[s]) min_r[s] = r; if (r > max_r[s]) max_r[s] = r;
            if (g < min_g[s]) min_g[s] = g; if (g > max_g[s]) max_g[s] = g;
            if (b < min_b[s]) min_b[s] = b; if (b > max_b[s]) max_b[s] = b;
            if (a < min_a[s]) min_a[s] = a; if (a > max_a[s]) max_a[s] = a;
        }

        // Skip partitions with empty subsets
        bool skip = false;
        for (uint32_t s = 0; s < total_subsets; s++)
            if (counts[s] == 0) { skip = true; break; }

        if (skip) { vol_scores[p] = UINT64_MAX; continue; }

        uint64_t vol = 0;
        for (uint32_t s = 0; s < total_subsets; s++)
        {
            vol += (uint32_t)(max_r[s] - min_r[s]) +
                   (uint32_t)(max_g[s] - min_g[s]) +
                   (uint32_t)(max_b[s] - min_b[s]);
            // FIX T (block-adaptive partition estimation): include alpha range
            // in volume score for bimodal alpha blocks. Bimodal blocks have
            // alpha 0+255 mix; partitions that mix alpha=0 with alpha=255 in
            // the same subset get a large alpha range contribution, naturally
            // downranking them. Partitions that separate alpha=0 from alpha=255
            // get alpha range 0 per subset, ranking them higher. This finds
            // alpha-separating partitions that the RGB-only volume score misses,
            // reducing the worst-case alpha error (was 161) on bimodal blocks.
            if (alpha_aware_volume)
                vol += (uint32_t)(max_a[s] - min_a[s]);
        }

        vol_scores[p] = vol;
    }

    // Stage 2: extract top PREFILTER_MAX candidates by volume score
    // Simple insertion sort into a small array
    struct { uint32_t index; uint64_t score; } top[64];
    uint32_t num_top = 0;
    for (uint32_t p = 0; p < total_partitions; p++)
    {
        if (vol_scores[p] == UINT64_MAX) continue;
        if (num_top < PREFILTER_MAX)
        {
            // Insert in sorted position
            int pos = (int)num_top;
            for (int j = 0; j < (int)num_top; j++)
            {
                if (vol_scores[p] < top[j].score) { pos = j; break; }
            }
            for (int j = (int)num_top; j > pos; j--)
                top[j] = top[j - 1];
            top[pos].index = p;
            top[pos].score = vol_scores[p];
            num_top++;
        }
        else if (vol_scores[p] < top[PREFILTER_MAX - 1].score)
        {
            // Replace worst entry
            int pos = (int)PREFILTER_MAX;
            for (int j = 0; j < (int)PREFILTER_MAX; j++)
            {
                if (vol_scores[p] < top[j].score) { pos = j; break; }
            }
            for (int j = PREFILTER_MAX - 1; j > pos; j--)
                top[j] = top[j - 1];
            top[pos].index = p;
            top[pos].score = vol_scores[p];
        }
    }

    // ============================================================
    // PHASE 2: Compactness + Overlap sort
    // ============================================================
    uint32_t num_p2 = num_top;
    phase2_candidate p2[64];

    for (uint32_t ti = 0; ti < num_top; ti++)
    {
        uint32_t p = top[ti].index;
        int min_r[3] = {255, 255, 255}, max_r[3] = {0, 0, 0};
        int min_g[3] = {255, 255, 255}, max_g[3] = {0, 0, 0};
        int min_b[3] = {255, 255, 255}, max_b[3] = {0, 0, 0};
        int min_a[3] = {255, 255, 255}, max_a[3] = {0, 0, 0};

        for (uint32_t s = 0; s < total_subsets; s++)
        {
            int cnt = g_bc7_subset_counts[p][s];
            const int *pix = g_bc7_subset_pixels[p][s];
            for (int j = 0; j < cnt; j++)
            {
                int idx = pix[j];
                int r = pPixels[idx].m_c[0], g = pPixels[idx].m_c[1], b = pPixels[idx].m_c[2];
                int a = pPixels[idx].m_c[3];
                if (r < min_r[s]) min_r[s] = r; if (r > max_r[s]) max_r[s] = r;
                if (g < min_g[s]) min_g[s] = g; if (g > max_g[s]) max_g[s] = g;
                if (b < min_b[s]) min_b[s] = b; if (b > max_b[s]) max_b[s] = b;
                if (a < min_a[s]) min_a[s] = a; if (a > max_a[s]) max_a[s] = a;
            }
        }

        uint64_t compactness = 0;
        for (uint32_t s = 0; s < total_subsets; s++)
        {
            compactness += (uint32_t)(max_r[s] - min_r[s]) +
                           (uint32_t)(max_g[s] - min_g[s]) +
                           (uint32_t)(max_b[s] - min_b[s]);
            // FIX T: alpha-aware compactness for bimodal blocks
            if (alpha_aware_volume)
                compactness += (uint32_t)(max_a[s] - min_a[s]);
        }

        uint64_t overlap = 0;
        for (uint32_t s0 = 0; s0 < total_subsets; s0++)
        {
            for (uint32_t s1 = s0 + 1; s1 < total_subsets; s1++)
            {
                int or_ = max(0, min(max_r[s0], max_r[s1]) - max(min_r[s0], min_r[s1]));
                int og = max(0, min(max_g[s0], max_g[s1]) - max(min_g[s0], min_g[s1]));
                int ob = max(0, min(max_b[s0], max_b[s1]) - max(min_b[s0], min_b[s1]));
                overlap += (uint32_t)(or_ + og + ob);
                // FIX T: alpha-aware overlap
                if (alpha_aware_volume)
                {
                    int oa = max(0, min(max_a[s0], max_a[s1]) - max(min_a[s0], min_a[s1]));
                    overlap += (uint32_t)oa;
                }
            }
        }

        p2[ti].partition = p;
        p2[ti].score = compactness * 2 - overlap;
        for (uint32_t s = 0; s < total_subsets; s++)
        {
            p2[ti].cached_min_r[s] = min_r[s]; p2[ti].cached_max_r[s] = max_r[s];
            p2[ti].cached_min_g[s] = min_g[s]; p2[ti].cached_max_g[s] = max_g[s];
            p2[ti].cached_min_b[s] = min_b[s]; p2[ti].cached_max_b[s] = max_b[s];
        }
    }

    // Sort by score ascending (best first), partition index tiebreaker
    for (uint32_t i = 1; i < num_p2; i++)
    {
        phase2_candidate tmp = p2[i];
        uint32_t j = i;
        while (j > 0 && (p2[j-1].score > tmp.score ||
               (p2[j-1].score == tmp.score && p2[j-1].partition > tmp.partition)))
        {
            p2[j] = p2[j-1];
            j--;
        }
        p2[j] = tmp;
    }


    // ============================================================
    // PASS 2: Expensive estimation only on pre-filtered candidates
    // ============================================================
    color_cell_compressor_params params;
    color_cell_compressor_params_clear(&params);

    params.m_pSelector_weights = (g_bc7_color_index_bitcount[mode] == 2) ?
        g_bc7_weights2 : g_bc7_weights3;
    params.m_num_selector_weights = 1 << g_bc7_color_index_bitcount[mode];

    memcpy(params.m_weights, pComp_params->m_weights, sizeof(params.m_weights));

    if (mode >= 6)
    {
        params.m_weights[0] *= pComp_params->m_alpha_settings.m_mode67_error_weight_mul[0];
        params.m_weights[1] *= pComp_params->m_alpha_settings.m_mode67_error_weight_mul[1];
        params.m_weights[2] *= pComp_params->m_alpha_settings.m_mode67_error_weight_mul[2];
        params.m_weights[3] *= pComp_params->m_alpha_settings.m_mode67_error_weight_mul[3];
    }

    // FIX T (block-adaptive partition estimation): for bimodal alpha blocks,
    // boost alpha weight 2x in the partition estimation pass. This ranks
    // alpha-separating partitions higher in the expensive Pass 2 estimation,
    // complementing the alpha-aware volume/compactness pre-filter in Pass 1.
    // The actual block encoding (in handle_alpha_block's Mode 7 loop) still
    // uses unboosted weights + Fix L's green 2x — only the PARTITION
    // ESTIMATION is alpha-aware, so we don't repeat Fix M's R/B regression
    // (Fix M boosted alpha in the final encoding, causing the encoder to
    // over-prioritize alpha at the cost of RGB).
    if (alpha_aware_volume)
    {
        params.m_weights[3] *= 2;
    }

    params.m_perceptual = pComp_params->m_perceptual;

    int32_t num_solutions = 0;

    for (uint32_t ti = 0; ti < num_p2; ti++)
    {
        uint32_t partition = p2[ti].partition;
        const int * pPartition = (total_subsets == 3) ?
            &g_bc7_partition3[partition * 16] : &g_bc7_partition2[partition * 16];

        color_quad_i subset_colors[3][16];
        uint32_t subset_total_colors[3];
        subset_total_colors[0] = 0;
        subset_total_colors[1] = 0;
        subset_total_colors[2] = 0;

        for (uint32_t index = 0; index < 16; index++)
        {
            const uint32_t p = pPartition[index];
            subset_colors[p][subset_total_colors[p]] = pPixels[index];
            subset_total_colors[p]++;
        }

        uint64_t total_subset_err = 0;
        uint64_t worst_err = (num_solutions >= max_solutions) ?
            pSolutions[max_solutions - 1].m_err : UINT64_MAX;

        for (uint32_t subset = 0; subset < total_subsets; subset++)
        {
            if (total_subset_err >= worst_err) break;
            uint64_t subset_budget = worst_err - total_subset_err;
            uint64_t err;
            if (mode == 7)
                err = color_cell_compression_est_mode7(mode, &params, subset_budget,
                    subset_total_colors[subset], &subset_colors[subset][0]);
            else
                err = color_cell_compression_est(mode, &params, subset_budget,
                    subset_total_colors[subset], &subset_colors[subset][0]);

            total_subset_err += err;
            if (err >= subset_budget) break;
        } // subset

        // Insert into sorted solutions (same as original)
        int32_t i;
        for (i = 0; i < num_solutions; i++)
        {
            if (total_subset_err < pSolutions[i].m_err)
                break;
        }

        if (i < num_solutions)
        {
            int32_t solutions_to_move = (max_solutions - 1) - i;
            int32_t num_elements_at_i = num_solutions - i;
            if (solutions_to_move > num_elements_at_i)
                solutions_to_move = num_elements_at_i;

            assert(((i + 1) + solutions_to_move) <= max_solutions);
            assert((i + solutions_to_move) <= num_solutions);

            for (int32_t j = solutions_to_move - 1; j >= 0; --j)
                pSolutions[i + j + 1] = pSolutions[i + j];
        }

        if (num_solutions < max_solutions)
            num_solutions++;

        if (i < num_solutions)
        {
            pSolutions[i].m_err = total_subset_err;
            pSolutions[i].m_index = partition;
        }

    } // ti (pre-filtered candidates only)

    return min(num_solutions, orig_max_solutions);
}

static inline void set_block_bits(uint8_t *pBytes, uint32_t val, uint32_t num_bits,  uint32_t * pCur_ofs)
{
                assert(num_bits < 32);
                uint32_t limit = 1U << num_bits;
                assert(val < limit);
                                
                while (num_bits)
                {
                                const uint32_t n = minimumu(8 - (*pCur_ofs & 7), num_bits);

// #pragma ignore warning(perf)  [ISPC pragma, no-op in C++]
                                pBytes[*pCur_ofs >> 3] |= (uint8_t)(val << (*pCur_ofs & 7));

                                val >>= n;
                                num_bits -= n;
                                *pCur_ofs += n;
                }

                assert(*pCur_ofs <= 128);
}

struct bc7_optimization_results
{
                uint32_t m_mode;
                uint32_t m_partition;
                int m_selectors[16];
                int m_alpha_selectors[16];
                color_quad_i m_low[3];
                color_quad_i m_high[3];
                uint32_t m_pbits[3][2];
                uint32_t m_rotation;
                uint32_t m_index_selector;
};

// Fast path: encode a flat (all-same-pixel) block as BC7 Mode 5.
// Based on: https://fgiesen.wordpress.com/2024/11/03/bc7-optimal-solid-color-blocks/
//
// Mode 5 gives 7-bit color endpoints + 8-bit alpha endpoints, separate color/alpha selectors.
// For the alpha channel: just send the 8-bit value directly as endpoint 0, all alpha indices = 0.
// For RGB: use index 1 (interpolation factor 21/64) with the formula:
//   e0 = target >> 1
//   e1 = ((target < 128) ? (target + 1) : (target - 1)) >> 1
// This reproduces every 8-bit value exactly via BC7's interpolation:
//   (43*expand7(e0) + 21*expand7(e1) + 32) >> 6 == target
//
// Mode 5 bit layout (128 bits):
//   Bits 0-5:   mode marker (bit 5 set)
//   Bits 6-7:   rotation = 0
//   (no partition bits - 1 subset)
//   (no index selector bit for mode 5 - it's only for mode 4)
//   Bits 8-14:  low R  (7 bits)   Bits 15-21: high R (7 bits)
//   Bits 22-28: low G  (7 bits)   Bits 29-35: high G (7 bits)
//   Bits 36-42: low B  (7 bits)   Bits 43-49: high B (7 bits)
//   Bits 50-57: low A  (8 bits)   Bits 58-65: high A (8 bits)
//   (no p-bits for mode 5)
//   Color indices: 2 bits x 16 pixels = 32 bits (index 0 = anchor, gets 1 bit)
//   Alpha indices: 2 bits x 16 pixels = 32 bits (index 0 = anchor, gets 1 bit)
static void encode_bc7_flat_block_mode5(void *pBlock, uint32_t r, uint32_t g, uint32_t b, uint32_t a)
{
                // Compute optimal 7-bit endpoint pairs for each RGB channel
                uint32_t r0 = r >> 1;
                uint32_t r1 = ((r < 128) ? (r + 1) : (r - 1)) >> 1;
                uint32_t g0 = g >> 1;
                uint32_t g1 = ((g < 128) ? (g + 1) : (g - 1)) >> 1;
                uint32_t b0 = b >> 1;
                uint32_t b1 = ((b < 128) ? (b + 1) : (b - 1)) >> 1;

                uint8_t block[16];
                memset(block, 0, 16);
                uint32_t cur_bit_ofs = 0;

                // Mode 5 marker
                set_block_bits(block, 1u << 5, 6, &cur_bit_ofs);
                // Rotation = 0 (no channel swap)
                set_block_bits(block, 0, 2, &cur_bit_ofs);
                // Color endpoints: R low(7) R high(7) G low(7) G high(7) B low(7) B high(7)
                set_block_bits(block, r0, 7, &cur_bit_ofs);
                set_block_bits(block, r1, 7, &cur_bit_ofs);
                set_block_bits(block, g0, 7, &cur_bit_ofs);
                set_block_bits(block, g1, 7, &cur_bit_ofs);
                set_block_bits(block, b0, 7, &cur_bit_ofs);
                set_block_bits(block, b1, 7, &cur_bit_ofs);
                // Alpha endpoints: A low(8) A high(8) - high doesn't matter since all alpha indices = 0
                set_block_bits(block, a, 8, &cur_bit_ofs);
                set_block_bits(block, a, 8, &cur_bit_ofs);
                // No p-bits for mode 5

                // Color indices: all = 1, except pixel 0 (anchor) which gets 1 bit (high bit = 0)
                // Anchor pixel (index 0) gets color_index_bits - 1 = 1 bit, value must have MSB = 0
                set_block_bits(block, 1, 1, &cur_bit_ofs); // pixel 0: color index 1, stored as 1-bit (MSB=0, just the LSB)
                for (int i = 1; i < 16; i++)
                                set_block_bits(block, 1, 2, &cur_bit_ofs); // pixels 1-15: color index 1

                // Alpha indices: all = 0, except pixel 0 (anchor) which gets alpha_index_bits - 1 = 1 bit
                set_block_bits(block, 0, 1, &cur_bit_ofs); // pixel 0: alpha index 0 (1 bit)
                for (int i = 1; i < 16; i++)
                                set_block_bits(block, 0, 2, &cur_bit_ofs); // pixels 1-15: alpha index 0

                memcpy(pBlock, block, 16);
}

static void encode_bc7_nearflat_block_mode5(
        void *pBlock,
        const color_quad_i *pPixels,
        uint32_t rMin, uint32_t rMax,
        uint32_t gMin, uint32_t gMax,
        uint32_t bMin, uint32_t bMax,
        uint32_t aMin, uint32_t aMax)
{
        // 7-bit endpoints for RGB
        uint32_t r0 = rMin >> 1, r1 = rMax >> 1;
        uint32_t g0 = gMin >> 1, g1 = gMax >> 1;
        uint32_t b0 = bMin >> 1, b1 = bMax >> 1;
        // 8-bit endpoints for alpha
        uint32_t a0 = aMin, a1 = aMax;

        // Reconstruct the 4 interpolation levels (8-bit) for each channel
        uint32_t rL[4], gL[4], bL[4], aL[4];
        uint32_t r0_8 = (r0 << 1) | (r0 >> 6);  // expand 7->8
        uint32_t r1_8 = (r1 << 1) | (r1 >> 6);
        uint32_t g0_8 = (g0 << 1) | (g0 >> 6);
        uint32_t g1_8 = (g1 << 1) | (g1 >> 6);
        uint32_t b0_8 = (b0 << 1) | (b0 >> 6);
        uint32_t b1_8 = (b1 << 1) | (b1 >> 6);
        // Mode 5 weights: 0, 21/64, 43/64, 64/64
        rL[0]=r0_8; rL[1]=(r0_8*43+r1_8*21+32)>>6; rL[2]=(r0_8*21+r1_8*43+32)>>6; rL[3]=r1_8;
        gL[0]=g0_8; gL[1]=(g0_8*43+g1_8*21+32)>>6; gL[2]=(g0_8*21+g1_8*43+32)>>6; gL[3]=g1_8;
        bL[0]=b0_8; bL[1]=(b0_8*43+b1_8*21+32)>>6; bL[2]=(b0_8*21+b1_8*43+32)>>6; bL[3]=b1_8;
        aL[0]=a0;   aL[1]=(a0*43+a1*21+32)>>6;     aL[2]=(a0*21+a1*43+32)>>6;     aL[3]=a1;

        // Assign each pixel to nearest interpolation index
        int color_idx[16], alpha_idx[16];
        for (int i = 0; i < 16; i++) {
                uint32_t pr = pPixels[i].m_c[0], pg = pPixels[i].m_c[1];
                uint32_t pb = pPixels[i].m_c[2], pa = pPixels[i].m_c[3];
                int bestC = 0, bestA = 0, bestCE = 0x7FFFFFFF, bestAE = 0x7FFFFFFF;
                for (int k = 0; k < 4; k++) {
                        int dr=(int)rL[k]-(int)pr, dg=(int)gL[k]-(int)pg, db=(int)bL[k]-(int)pb;
                        int ce = dr*dr+dg*dg+db*db;
                        if (ce < bestCE) { bestCE = ce; bestC = k; }
                        int da = (int)aL[k]-(int)pa;
                        int ae = da*da;
                        if (ae < bestAE) { bestAE = ae; bestA = k; }
                }
                color_idx[i] = bestC;
                alpha_idx[i] = bestA;
        }

        // --- Pack the 128-bit Mode 5 block ---
        uint8_t blk[16];
        memset(blk, 0, 16);
        uint32_t bit = 0;
        auto setBits = [&](uint32_t val, uint32_t n) {
                for (uint32_t i = 0; i < n; i++, bit++)
                        blk[bit>>3] |= ((val>>i)&1) << (bit&7);
        };

        setBits(1<<5, 8);         // mode = 5  (bit 5 set)
        setBits(0, 2);            // rotation = 0
        setBits(0, 1);            // index_selection = 0

        // Endpoints: R0,R1,G0,G1,B0,B1 (7 bits each), A0,A1 (8 bits each)
        setBits(r0,7); setBits(r1,7);
        setBits(g0,7); setBits(g1,7);
        setBits(b0,7); setBits(b1,7);
        setBits(a0,8); setBits(a1,8);

        // Color indices: 2 bits × 16 (anchor pixel 0 gets 1 bit)
        for (int i = 0; i < 16; i++)
                setBits(color_idx[i], (i==0)?1:2);
        // Alpha indices: 2 bits × 16 (anchor pixel 0 gets 1 bit)
        for (int i = 0; i < 16; i++)
                setBits(alpha_idx[i], (i==0)?1:2);

        memcpy(pBlock, blk, 16);
}

static void encode_bc7_block(void *pBlock, const  bc7_optimization_results * pResults)
{
                const uint32_t best_mode = pResults->m_mode;

// #pragma ignore warning(perf)  [ISPC pragma, no-op in C++]
                const uint32_t total_subsets = g_bc7_num_subsets[best_mode];

// #pragma ignore warning(perf)  [ISPC pragma, no-op in C++]
                const uint32_t total_partitions = 1 << g_bc7_partition_bits[best_mode];

                const int *pPartition;
                if (total_subsets == 1)
                                pPartition = &g_bc7_partition1[0];
                else if (total_subsets == 2)
                                pPartition = &g_bc7_partition2[pResults->m_partition * 16];
                else
                                pPartition = &g_bc7_partition3[pResults->m_partition * 16];

                int color_selectors[16];
                for ( int i = 0; i < 16; i++)
                                color_selectors[i] = pResults->m_selectors[i];

                int alpha_selectors[16];
                for ( int i = 0; i < 16; i++)
                                alpha_selectors[i] = pResults->m_alpha_selectors[i];

                color_quad_i low[3], high[3];
                low[0] = pResults->m_low[0];
                low[1] = pResults->m_low[1];
                low[2] = pResults->m_low[2];

                high[0] = pResults->m_high[0];
                high[1] = pResults->m_high[1];
                high[2] = pResults->m_high[2];
                
                uint32_t pbits[3][2];
                for ( int i = 0; i < 3; i++)
                {
                                pbits[i][0] = pResults->m_pbits[i][0];
                                pbits[i][1] = pResults->m_pbits[i][1];
                }

                int anchor[3];
                anchor[0] = -1;
                anchor[1] = -1;
                anchor[2] = -1;

                for ( uint32_t k = 0; k < total_subsets; k++)
                {
                                uint32_t anchor_index = 0;
                                if (k)
                                {
                                                if ((total_subsets == 3) && (k == 1))
                                                {
// #pragma ignore warning(perf)  [ISPC pragma, no-op in C++]
                                                                anchor_index = g_bc7_table_anchor_index_third_subset_1[pResults->m_partition];
                                                }
                                                else if ((total_subsets == 3) && (k == 2))
                                                {
// #pragma ignore warning(perf)  [ISPC pragma, no-op in C++]
                                                                anchor_index = g_bc7_table_anchor_index_third_subset_2[pResults->m_partition];
                                                }
                                                else
                                                {
// #pragma ignore warning(perf)  [ISPC pragma, no-op in C++]
                                                                anchor_index = g_bc7_table_anchor_index_second_subset[pResults->m_partition];
                                                }
                                }

                                anchor[k] = anchor_index;

                                const uint32_t color_index_bits = get_bc7_color_index_size(best_mode, pResults->m_index_selector);
                                const uint32_t num_color_indices = 1 << color_index_bits;

// #pragma ignore warning(perf)  [ISPC pragma, no-op in C++]
                                if (color_selectors[anchor_index] & (num_color_indices >> 1))
                                {
                                                for ( uint32_t i = 0; i < 16; i++)
                                                {
// #pragma ignore warning(perf)  [ISPC pragma, no-op in C++]
                                                                if (pPartition[i] == k)
                                                                                color_selectors[i] = (num_color_indices - 1) - color_selectors[i];
                                                }

                                                if (get_bc7_mode_has_seperate_alpha_selectors(best_mode))
                                                {
                                                                for ( uint32_t q = 0; q < 3; q++)
                                                                {
                                                                                int t = low[k].m_c[q];
                                                                                low[k].m_c[q] = high[k].m_c[q];
                                                                                high[k].m_c[q] = t;
                                                                }
                                                }
                                                else
                                                {
                                                                color_quad_i tmp = low[k];
                                                                low[k] = high[k];
                                                                high[k] = tmp;
                                                }

// #pragma ignore warning(perf)  [ISPC pragma, no-op in C++]
                                                if (!g_bc7_mode_has_shared_p_bits[best_mode])
                                                {
                                                                uint32_t t = pbits[k][0];
                                                                pbits[k][0] = pbits[k][1];
                                                                pbits[k][1] = t;
                                                }
                                }

                                if (get_bc7_mode_has_seperate_alpha_selectors(best_mode))
                                {
                                                const uint32_t alpha_index_bits = get_bc7_alpha_index_size(best_mode, pResults->m_index_selector);
                                                const uint32_t num_alpha_indices = 1 << alpha_index_bits;

// #pragma ignore warning(perf)  [ISPC pragma, no-op in C++]
                                                if (alpha_selectors[anchor_index] & (num_alpha_indices >> 1))
                                                {
                                                                for ( uint32_t i = 0; i < 16; i++)
                                                                {
// #pragma ignore warning(perf)  [ISPC pragma, no-op in C++]
                                                                                if (pPartition[i] == k)
                                                                                                alpha_selectors[i] = (num_alpha_indices - 1) - alpha_selectors[i];
                                                                }

                                                                int t = low[k].m_c[3];
                                                                low[k].m_c[3] = high[k].m_c[3];
                                                                high[k].m_c[3] = t;
                                                }
                                }
                }

                uint8_t *pBlock_bytes = (uint8_t *)(pBlock);
                memset(pBlock_bytes, 0, BC7E_BLOCK_SIZE);

                uint32_t cur_bit_ofs = 0;
                                
                set_block_bits(pBlock_bytes, 1 << best_mode, best_mode + 1, &cur_bit_ofs);

                if ((best_mode == 4) || (best_mode == 5))
                                set_block_bits(pBlock_bytes, pResults->m_rotation, 2, &cur_bit_ofs);

                if (best_mode == 4)
                                set_block_bits(pBlock_bytes, pResults->m_index_selector, 1, &cur_bit_ofs);

                if (total_partitions > 1)
                                set_block_bits(pBlock_bytes, pResults->m_partition, (total_partitions == 64) ? 6 : 4, &cur_bit_ofs);

                const uint32_t total_comps = (best_mode >= 4) ? 4 : 3;
                for ( uint32_t comp = 0; comp < total_comps; comp++)
                {
                                for ( uint32_t subset = 0; subset < total_subsets; subset++)
                                {
                                                set_block_bits(pBlock_bytes, low[subset].m_c[comp], (comp == 3) ? g_bc7_alpha_precision_table[best_mode] : g_bc7_color_precision_table[best_mode], &cur_bit_ofs);
                                                set_block_bits(pBlock_bytes, high[subset].m_c[comp], (comp == 3) ? g_bc7_alpha_precision_table[best_mode] : g_bc7_color_precision_table[best_mode], &cur_bit_ofs);
                                }
                }

// #pragma ignore warning(perf)  [ISPC pragma, no-op in C++]
                if (g_bc7_mode_has_p_bits[best_mode])
                {
                                for ( uint32_t subset = 0; subset < total_subsets; subset++)
                                {
                                                set_block_bits(pBlock_bytes, pbits[subset][0], 1, &cur_bit_ofs);
// #pragma ignore warning(perf)  [ISPC pragma, no-op in C++]
                                                if (!g_bc7_mode_has_shared_p_bits[best_mode])
                                                                set_block_bits(pBlock_bytes, pbits[subset][1], 1, &cur_bit_ofs);
                                }
                }

                for ( uint32_t y = 0; y < 4; y++)
                {
                                for ( uint32_t x = 0; x < 4; x++)
                                {
                                                 int idx = x + y * 4;

                                                uint32_t n = pResults->m_index_selector ? get_bc7_alpha_index_size(best_mode, pResults->m_index_selector) : get_bc7_color_index_size(best_mode, pResults->m_index_selector);

                                                if ((idx == anchor[0]) || (idx == anchor[1]) || (idx == anchor[2]))
                                                                n--;

                                                set_block_bits(pBlock_bytes, pResults->m_index_selector ? alpha_selectors[idx] : color_selectors[idx], n, &cur_bit_ofs);
                                }
                }

                if (get_bc7_mode_has_seperate_alpha_selectors(best_mode))
                {
                                for ( uint32_t y = 0; y < 4; y++)
                                {
                                                for ( uint32_t x = 0; x < 4; x++)
                                                {
                                                                int idx = x + y * 4;

                                                                uint32_t n = pResults->m_index_selector ? get_bc7_color_index_size(best_mode, pResults->m_index_selector) : get_bc7_alpha_index_size(best_mode, pResults->m_index_selector);

                                                                if ((idx == anchor[0]) || (idx == anchor[1]) || (idx == anchor[2]))
                                                                                n--;

                                                                set_block_bits(pBlock_bytes, pResults->m_index_selector ? color_selectors[idx] : alpha_selectors[idx], n, &cur_bit_ofs);
                                                }
                                }
                }

                assert(cur_bit_ofs == 128);
}

static inline void encode_bc7_block_mode6(void *pBlock,  bc7_optimization_results * pResults)
{
                color_quad_i low, high;
                uint32_t pbits[2];
                                
                uint32_t invert_selectors = 0;
                if (pResults->m_selectors[0] & 8)
                {
                                invert_selectors = 15;
                                                                                                                
                                low = pResults->m_high[0];
                                high = pResults->m_low[0];

                                pbits[0] = pResults->m_pbits[0][1];
                                pbits[1] = pResults->m_pbits[0][0];
                }
                else
                {
                                low = pResults->m_low[0];
                                high = pResults->m_high[0];

                                pbits[0] = pResults->m_pbits[0][0];
                                pbits[1] = pResults->m_pbits[0][1];
                }

                uint64_t l = 0, h = 0;

                l = 1 << 6;

                l |= (low.m_c[0] << 7);
                l |= (high.m_c[0] << 14);

                l |= (low.m_c[1] << 21);
                l |= ((uint64_t)high.m_c[1] << 28);

                l |= ((uint64_t)low.m_c[2] << 35);
                l |= ((uint64_t)high.m_c[2] << 42);

                l |= ((uint64_t)low.m_c[3] << 49);
                l |= ((uint64_t)high.m_c[3] << 56);

                l |= ((uint64_t)pbits[0] << 63);
                                
                h = pbits[1];
                                
                h |= ((invert_selectors ^ pResults->m_selectors[0]) << 1);

                // TODO: Just invert all these bits in one single operation, not as individual
                h |= ((invert_selectors ^ pResults->m_selectors[1]) << 4);
                h |= ((invert_selectors ^ pResults->m_selectors[2]) << 8);
                h |= ((invert_selectors ^ pResults->m_selectors[3]) << 12);
                h |= ((invert_selectors ^ pResults->m_selectors[4]) << 16);
                
                h |= ((invert_selectors ^ pResults->m_selectors[5]) << 20);
                h |= ((invert_selectors ^ pResults->m_selectors[6]) << 24);
                h |= ((invert_selectors ^ pResults->m_selectors[7]) << 28);
                h |= ((uint64_t)(invert_selectors ^ pResults->m_selectors[8]) << 32);

                h |= ((uint64_t)(invert_selectors ^ pResults->m_selectors[9]) << 36);
                h |= ((uint64_t)(invert_selectors ^ pResults->m_selectors[10]) << 40);
                h |= ((uint64_t)(invert_selectors ^ pResults->m_selectors[11]) << 44);
                h |= ((uint64_t)(invert_selectors ^ pResults->m_selectors[12]) << 48);

                h |= ((uint64_t)(invert_selectors ^ pResults->m_selectors[13]) << 52);
                h |= ((uint64_t)(invert_selectors ^ pResults->m_selectors[14]) << 56);
                h |= ((uint64_t)(invert_selectors ^ pResults->m_selectors[15]) << 60);

// #pragma ignore warning(perf)  [ISPC pragma, no-op in C++]
                ((uint64_t *)(pBlock))[0] = l;

// #pragma ignore warning(perf)  [ISPC pragma, no-op in C++]
                ((uint64_t *)(pBlock))[1] = h;
}

static void handle_alpha_block_mode4(const  color_quad_i * pPixels, const  bc7e_compress_block_params * pComp_params,  color_cell_compressor_params * pParams, uint32_t lo_a, uint32_t hi_a, 
                 bc7_optimization_results * pOpt_results4,  uint64_t * pMode4_err)
{
                pParams->m_has_alpha = false;
                pParams->m_comp_bits = 5;
                pParams->m_has_pbits = false;
                pParams->m_endpoints_share_pbit = false;                                
                pParams->m_perceptual = pComp_params->m_perceptual;

                for ( uint32_t index_selector = 0; index_selector < 2; index_selector++)
                {
                                if ((pComp_params->m_mode4_index_mask & (1 << index_selector)) == 0)
                                                continue;

                                if (index_selector)
                                {
                                                pParams->m_pSelector_weights = g_bc7_weights3;
                                                pParams->m_pSelector_weightsx = (const vec4F * )&g_bc7_weights3x[0];
                                                pParams->m_num_selector_weights = 8;
                                }
                                else
                                {
                                                pParams->m_pSelector_weights = g_bc7_weights2;
                                                pParams->m_pSelector_weightsx = (const vec4F * )&g_bc7_weights2x[0];
                                                pParams->m_num_selector_weights = 4;
                                }
                                                                                                                                
                                color_cell_compressor_results results;
                                
                                int selectors[16];
                                results.m_pSelectors = selectors;

                                int selectors_temp[16];
                                results.m_pSelectors_temp = selectors_temp;
                                                                
                                uint64_t trial_err = color_cell_compression(4, pParams, &results, pComp_params, 16, pPixels, true);
                                assert(trial_err == results.m_best_overall_err);

                                uint32_t la = minimumi((lo_a + 2) >> 2, 63);
                                uint32_t ha = minimumi((hi_a + 2) >> 2, 63);

                                if (la == ha)
                                {
                                                if (lo_a != hi_a)
                                                {
                                                                if (ha != 63)
                                                                                ha++;
                                                                else if (la != 0)
                                                                                la--;
                                                }
                                }

                                uint64_t best_alpha_err = UINT64_MAX;
                                uint32_t best_la = 0, best_ha = 0;
                                int best_alpha_selectors[16];
                                                                                                
                                for ( int32_t pass = 0; pass < 2; pass++)
                                {
                                                int32_t vals[8];

                                                if (index_selector == 0)
                                                {
                                                                vals[0] = (la << 2) | (la >> 4);
                                                                vals[7] = (ha << 2) | (ha >> 4);

                                                                for ( uint32_t i = 1; i < 7; i++)
                                                                                vals[i] = (vals[0] * (64 - g_bc7_weights3[i]) + vals[7] * g_bc7_weights3[i] + 32) >> 6;
                                                }
                                                else
                                                {
                                                                vals[0] = (la << 2) | (la >> 4);
                                                                vals[3] = (ha << 2) | (ha >> 4);

                                                                const  int32_t w_s1 = 21, w_s2 = 43;
                                                                vals[1] = (vals[0] * (64 - w_s1) + vals[3] * w_s1 + 32) >> 6;
                                                                vals[2] = (vals[0] * (64 - w_s2) + vals[3] * w_s2 + 32) >> 6;
                                                }

                                                uint64_t trial_alpha_err = 0;

                                                int trial_alpha_selectors[16];
                                                for ( uint32_t i = 0; i < 16; i++)
                                                {
                                                                const int32_t a = pPixels[i].m_c[3];

                                                                int s = 0;
                                                                int32_t be = iabs32(a - vals[0]);

                                                                int e = iabs32(a - vals[1]); if (e < be) { be = e; s = 1; }
                                                                e = iabs32(a - vals[2]); if (e < be) { be = e; s = 2; }
                                                                e = iabs32(a - vals[3]); if (e < be) { be = e; s = 3; }

                                                                if (index_selector == 0)
                                                                {
                                                                                e = iabs32(a - vals[4]); if (e < be) { be = e; s = 4; }
                                                                                e = iabs32(a - vals[5]); if (e < be) { be = e; s = 5; }
                                                                                e = iabs32(a - vals[6]); if (e < be) { be = e; s = 6; }
                                                                                e = iabs32(a - vals[7]); if (e < be) { be = e; s = 7; }
                                                                }

                                                                trial_alpha_err += (be * be) * pParams->m_weights[3];

                                                                trial_alpha_selectors[i] = s;
                                                }

                                                if (trial_alpha_err < best_alpha_err)
                                                {
                                                                best_alpha_err = trial_alpha_err;
                                                                best_la = la;
                                                                best_ha = ha;
                                                                for ( uint32_t i = 0; i < 16; i++)
                                                                                best_alpha_selectors[i] = trial_alpha_selectors[i];
                                                }

                                                if (pass == 0) 
                                                {
                                                                float xl, xh;
                                                                compute_least_squares_endpoints_a(16, trial_alpha_selectors, index_selector ? (const vec4F * )&g_bc7_weights2x[0] : (const vec4F * )&g_bc7_weights3x[0], &xl, &xh, pPixels);
                                                                if (xl > xh)
                                                                                swapf(&xl, &xh);
                                                                la = clampi((int)floor(xl * (63.0f / 255.0f) + .5f), 0, 63);
                                                                ha = clampi((int)floor(xh * (63.0f / 255.0f) + .5f), 0, 63);
                                                }
                                                                                                
                                } // pass

                                if (pComp_params->m_uber_level > 0)
                                {
                                                const  int D = min((int)pComp_params->m_uber_level, 3);
                                                for ( int ld = -D; ld <= D; ld++)
                                                {
                                                                for ( int hd = -D; hd <= D; hd++)
                                                                {
                                                                                la = ispc_clamp((int)best_la + ld, 0, 63);
                                                                                ha = ispc_clamp((int)best_ha + hd, 0, 63);
                                                                                
                                                                                int32_t vals[8];

                                                                                if (index_selector == 0)
                                                                                {
                                                                                                vals[0] = (la << 2) | (la >> 4);
                                                                                                vals[7] = (ha << 2) | (ha >> 4);

                                                                                                for ( uint32_t i = 1; i < 7; i++)
                                                                                                                vals[i] = (vals[0] * (64 - g_bc7_weights3[i]) + vals[7] * g_bc7_weights3[i] + 32) >> 6;
                                                                                }
                                                                                else
                                                                                {
                                                                                                vals[0] = (la << 2) | (la >> 4);
                                                                                                vals[3] = (ha << 2) | (ha >> 4);

                                                                                                const  int32_t w_s1 = 21, w_s2 = 43;
                                                                                                vals[1] = (vals[0] * (64 - w_s1) + vals[3] * w_s1 + 32) >> 6;
                                                                                                vals[2] = (vals[0] * (64 - w_s2) + vals[3] * w_s2 + 32) >> 6;
                                                                                }

                                                                                uint64_t trial_alpha_err = 0;

                                                                                int trial_alpha_selectors[16];
                                                                                for ( uint32_t i = 0; i < 16; i++)
                                                                                {
                                                                                                const int32_t a = pPixels[i].m_c[3];

                                                                                                int s = 0;
                                                                                                int32_t be = iabs32(a - vals[0]);

                                                                                                int e = iabs32(a - vals[1]); if (e < be) { be = e; s = 1; }
                                                                                                e = iabs32(a - vals[2]); if (e < be) { be = e; s = 2; }
                                                                                                e = iabs32(a - vals[3]); if (e < be) { be = e; s = 3; }

                                                                                                if (index_selector == 0)
                                                                                                {
                                                                                                                e = iabs32(a - vals[4]); if (e < be) { be = e; s = 4; }
                                                                                                                e = iabs32(a - vals[5]); if (e < be) { be = e; s = 5; }
                                                                                                                e = iabs32(a - vals[6]); if (e < be) { be = e; s = 6; }
                                                                                                                e = iabs32(a - vals[7]); if (e < be) { be = e; s = 7; }
                                                                                                }

                                                                                                trial_alpha_err += (be * be) * pParams->m_weights[3];

                                                                                                trial_alpha_selectors[i] = s;
                                                                                }

                                                                                if (trial_alpha_err < best_alpha_err)
                                                                                {
                                                                                                best_alpha_err = trial_alpha_err;
                                                                                                best_la = la;
                                                                                                best_ha = ha;
                                                                                                for ( uint32_t i = 0; i < 16; i++)
                                                                                                                best_alpha_selectors[i] = trial_alpha_selectors[i];
                                                                                }
                                                                
                                                                } // hd

                                                } // ld
                                }

                                trial_err += best_alpha_err;

                                if (trial_err < *pMode4_err)
                                {
                                                *pMode4_err = trial_err;

                                                pOpt_results4->m_mode = 4;
                                                pOpt_results4->m_index_selector = index_selector;
                                                pOpt_results4->m_rotation = 0;
                                                pOpt_results4->m_partition = 0;

                                                pOpt_results4->m_low[0] = results.m_low_endpoint;
                                                pOpt_results4->m_high[0] = results.m_high_endpoint;
                                                pOpt_results4->m_low[0].m_c[3] = best_la;
                                                pOpt_results4->m_high[0].m_c[3] = best_ha;

                                                for ( uint32_t i = 0; i < 16; i++)
                                                                pOpt_results4->m_selectors[i] = selectors[i];

                                                for ( uint32_t i = 0; i < 16; i++)
                                                                pOpt_results4->m_alpha_selectors[i] = best_alpha_selectors[i];
                                }

                } // index_selector
}

static void handle_alpha_block_mode5(const  color_quad_i * pPixels, const  bc7e_compress_block_params * pComp_params,  color_cell_compressor_params * pParams, uint32_t lo_a, uint32_t hi_a, 
                 bc7_optimization_results * pOpt_results5,  uint64_t * pMode5_err)
{
                pParams->m_pSelector_weights = g_bc7_weights2;
                pParams->m_pSelector_weightsx = (const vec4F * )&g_bc7_weights2x[0];
                pParams->m_num_selector_weights = 4;

                pParams->m_comp_bits = 7;
                pParams->m_has_alpha = false;
                pParams->m_has_pbits = false;
                pParams->m_endpoints_share_pbit = false;                                
                
                pParams->m_perceptual = pComp_params->m_perceptual;
                                
                color_cell_compressor_results results5;
                results5.m_pSelectors = pOpt_results5->m_selectors;

                int selectors_temp[16];
                results5.m_pSelectors_temp = selectors_temp;

                *pMode5_err = color_cell_compression(5, pParams, &results5, pComp_params, 16, pPixels, true);
                assert(*pMode5_err == results5.m_best_overall_err);

                pOpt_results5->m_low[0] = results5.m_low_endpoint;
                pOpt_results5->m_high[0] = results5.m_high_endpoint;

                if (lo_a == hi_a)
                {
                                pOpt_results5->m_low[0].m_c[3] = lo_a;
                                pOpt_results5->m_high[0].m_c[3] = hi_a;
                                for ( uint32_t i = 0; i < 16; i++)
                                                pOpt_results5->m_alpha_selectors[i] = 0;
                }
                else
                {
                                uint64_t mode5_alpha_err = UINT64_MAX;

                                for ( uint32_t pass = 0; pass < 2; pass++)
                                {
                                                int32_t vals[4];
                                                vals[0] = lo_a;
                                                vals[3] = hi_a;

                                                const  int32_t w_s1 = 21, w_s2 = 43;
                                                vals[1] = (vals[0] * (64 - w_s1) + vals[3] * w_s1 + 32) >> 6;
                                                vals[2] = (vals[0] * (64 - w_s2) + vals[3] * w_s2 + 32) >> 6;

                                                int trial_alpha_selectors[16];

                                                uint64_t trial_alpha_err = 0;
                                                for ( uint32_t i = 0; i < 16; i++)
                                                {
                                                                const int32_t a = pPixels[i].m_c[3];

                                                                int s = 0;
                                                                int32_t be = iabs32(a - vals[0]);
                                                                int e = iabs32(a - vals[1]); if (e < be) { be = e; s = 1; }
                                                                e = iabs32(a - vals[2]); if (e < be) { be = e; s = 2; }
                                                                e = iabs32(a - vals[3]); if (e < be) { be = e; s = 3; }

                                                                trial_alpha_selectors[i] = s;
                                                                                                                                
                                                                trial_alpha_err += (be * be) * pParams->m_weights[3];
                                                }

                                                if (trial_alpha_err < mode5_alpha_err)
                                                {
                                                                mode5_alpha_err = trial_alpha_err;
                                                                pOpt_results5->m_low[0].m_c[3] = lo_a;
                                                                pOpt_results5->m_high[0].m_c[3] = hi_a;
                                                                for ( uint32_t i = 0; i < 16; i++)
                                                                                pOpt_results5->m_alpha_selectors[i] = trial_alpha_selectors[i];
                                                }

                                                if (!pass)
                                                {
                                                                float xl, xh;
                                                                compute_least_squares_endpoints_a(16, trial_alpha_selectors, (const vec4F * )&g_bc7_weights2x[0], &xl, &xh, pPixels);

                                                                uint32_t new_lo_a = clampi((int)floor(xl + .5f), 0, 255);
                                                                uint32_t new_hi_a = clampi((int)floor(xh + .5f), 0, 255);
                                                                if (new_lo_a > new_hi_a)
                                                                                swapu(&new_lo_a, &new_hi_a);

                                                                if ((new_lo_a == lo_a) && (new_hi_a == hi_a))
                                                                                break;

                                                                lo_a = new_lo_a;
                                                                hi_a = new_hi_a;
                                                }
                                }

                                if (pComp_params->m_uber_level > 0)
                                {
                                                const  int D = min((int)pComp_params->m_uber_level, 3);
                                                for ( int ld = -D; ld <= D; ld++)
                                                {
                                                                for ( int hd = -D; hd <= D; hd++)
                                                                {
                                                                                lo_a = ispc_clamp((int)pOpt_results5->m_low[0].m_c[3] + ld, 0, 255);
                                                                                hi_a = ispc_clamp((int)pOpt_results5->m_high[0].m_c[3] + hd, 0, 255);
                                                                                
                                                                                int32_t vals[4];
                                                                                vals[0] = lo_a;
                                                                                vals[3] = hi_a;

                                                                                const  int32_t w_s1 = 21, w_s2 = 43;
                                                                                vals[1] = (vals[0] * (64 - w_s1) + vals[3] * w_s1 + 32) >> 6;
                                                                                vals[2] = (vals[0] * (64 - w_s2) + vals[3] * w_s2 + 32) >> 6;

                                                                                int trial_alpha_selectors[16];

                                                                                uint64_t trial_alpha_err = 0;
                                                                                for ( uint32_t i = 0; i < 16; i++)
                                                                                {
                                                                                                const int32_t a = pPixels[i].m_c[3];

                                                                                                int s = 0;
                                                                                                int32_t be = iabs32(a - vals[0]);
                                                                                                int e = iabs32(a - vals[1]); if (e < be) { be = e; s = 1; }
                                                                                                e = iabs32(a - vals[2]); if (e < be) { be = e; s = 2; }
                                                                                                e = iabs32(a - vals[3]); if (e < be) { be = e; s = 3; }

                                                                                                trial_alpha_selectors[i] = s;
                                                                                                                                
                                                                                                trial_alpha_err += (be * be) * pParams->m_weights[3];
                                                                                }

                                                                                if (trial_alpha_err < mode5_alpha_err)
                                                                                {
                                                                                                mode5_alpha_err = trial_alpha_err;
                                                                                                pOpt_results5->m_low[0].m_c[3] = lo_a;
                                                                                                pOpt_results5->m_high[0].m_c[3] = hi_a;
                                                                                                for ( uint32_t i = 0; i < 16; i++)
                                                                                                                pOpt_results5->m_alpha_selectors[i] = trial_alpha_selectors[i];
                                                                                }
                                                                
                                                                } // hd

                                                } // ld
                                }

                                *pMode5_err += mode5_alpha_err;
                }

                pOpt_results5->m_mode = 5;
                pOpt_results5->m_index_selector = 0;
                pOpt_results5->m_rotation = 0;
                pOpt_results5->m_partition = 0;
}

static void handle_alpha_block(void * pBlock, const  color_quad_i * pPixels, const  bc7e_compress_block_params * pComp_params,  color_cell_compressor_params * pParams, uint32_t lo_a, uint32_t hi_a, const bc7_block_stats * pStats, bool bimodal_alpha)
{
        pParams->m_perceptual = pComp_params->m_perceptual;
        bc7_optimization_results opt_results;
        memset(&opt_results, 0, sizeof(opt_results));
        uint64_t best_err = UINT64_MAX;

        // OPT-D (speed): use per-channel RGB min/max from pStats (computed
        // once in compute_block_stats) instead of re-scanning 16 pixels.
        const int lo_r = pStats->min_r, hi_r = pStats->max_r;
        const int lo_g = pStats->min_g, hi_g = pStats->max_g;
        const int lo_b = pStats->min_b, hi_b = pStats->max_b;

        // QuickBC7 alpha mode pruning
        const int alpha_range = (int)(hi_a - lo_a);
        // FIX N (residual green outline): outline blocks have very low alpha
        // variance (alpha_range <= 16) but may span full color range (e.g.
        // deep-blue G=0 next to bright-cyan G=85). Mode 7's partition
        // estimator often groups these dissimilar colors in the same subset,
        // causing green to drift up to G=12/24/73 -- the residual 'grayish
        // green on deep purple outline' bug. Tagging these blocks so all
        // modes can apply a green upweight (Fix N) to bias partition
        // selection toward green-preserving partitions.
        const bool outline_alpha_block = (alpha_range <= 16);
        const int ndc = pStats->num_distinct_colors;
        const int luma_range = pStats->max_luma - pStats->min_luma;

        // If very few distinct colors, multi-subset mode 7 is pointless.
        const bool skip_mode7 = (ndc <= 3) || (luma_range <= 16 && pStats->luma_var < 128);

        // Mode 4 - SKIP when color range exceeds 5-bit precision capacity
        const int color_range_r = (int)(hi_r - lo_r);
        const int color_range_g = (int)(hi_g - lo_g);
        const int color_range_b = (int)(hi_b - lo_b);
        const int max_color_range = maximumi(maximumi(color_range_r, color_range_g), color_range_b);
        const bool skip_mode4 = (max_color_range > 64); // 5-bit can't handle large ranges

        // Mode 4
        if (pComp_params->m_alpha_settings.m_use_mode4 && !skip_mode4)
        {
                 color_cell_compressor_params params4 = *pParams;
                const  int num_rotations = (pComp_params->m_perceptual || (!pComp_params->m_alpha_settings.m_use_mode4_rotation)) ? 1 : 4;
                for ( uint32_t rotation = 0; rotation < num_rotations; rotation++)
                {
                                if ((pComp_params->m_mode4_rotation_mask & (1 << rotation)) == 0) continue;
                                memcpy(params4.m_weights, pParams->m_weights, sizeof(params4.m_weights));
                                if (bimodal_alpha) params4.m_weights[1] *= 2; // FIX L: green upweight for bimodal
                                if (rotation) swapu(&params4.m_weights[rotation - 1], &params4.m_weights[3]);
                                                                                                
                                color_quad_i rot_pixels[16];
                                const  color_quad_i * pTrial_pixels = pPixels;
                                uint32_t trial_lo_a = lo_a, trial_hi_a = hi_a;
                                if (rotation)
                                {
                                                trial_lo_a = 255; trial_hi_a = 0;
                                                for ( uint32_t i = 0; i < 16; i++)
                                                {
                                                                color_quad_i c = pPixels[i];
                                                                swapi(&c.m_c[3], &c.m_c[rotation - 1]);
                                                                rot_pixels[i] = c;
                                                                trial_lo_a = minimumu(trial_lo_a, c.m_c[3]);
                                                                trial_hi_a = maximumu(trial_hi_a, c.m_c[3]);
                                                }
                                                pTrial_pixels = rot_pixels;
                                }

                                bc7_optimization_results trial_opt_results4;
                                uint64_t trial_mode4_err = best_err;
                                handle_alpha_block_mode4(pTrial_pixels, pComp_params, &params4, trial_lo_a, trial_hi_a, &trial_opt_results4, &trial_mode4_err);

                                if (trial_mode4_err < best_err)
                                {
                                                best_err = trial_mode4_err;
                                                opt_results.m_mode = 4;
                                                opt_results.m_index_selector = trial_opt_results4.m_index_selector;
                                                opt_results.m_rotation = rotation;
                                                opt_results.m_partition = 0;
                                                opt_results.m_low[0] = trial_opt_results4.m_low[0];
                                                opt_results.m_high[0] = trial_opt_results4.m_high[0];
                                                for ( uint32_t i = 0; i < 16; i++) opt_results.m_selectors[i] = trial_opt_results4.m_selectors[i];
                                                for ( uint32_t i = 0; i < 16; i++) opt_results.m_alpha_selectors[i] = trial_opt_results4.m_alpha_selectors[i];
                                }
                } // rotation
        }
        
        // Mode 6 - MUST REMAIN ACTIVE AS FALLBACK!
        // The original bimodal alpha fix is restored here to prevent PCA starvation.
        const bool near_opaque = (lo_a >= 240);
        // FIX I (Mode 6 expansion): originally Mode 6 was gated to
        // `!bimodal_alpha && near_opaque` (lo_a >= 240), which excluded all
        // mid-range alpha blocks (e.g. soft sprite edges with alpha sweeping
        // 50->200). Those blocks fell back to Mode 4 (8 alpha selectors, 5-bit
        // RGB) or Mode 5 (4 shared selectors, 7-bit RGB) or Mode 7 (4 shared,
        // 5-bit RGB+A) -- all of which sacrifice either alpha or RGB precision.
        // Mode 6 has 16 selectors (best alpha gradient coverage) AND 7-bit RGB
        // endpoints (good color precision), making it strictly better for
        // smooth alpha gradients. Expanding the gate to `!bimodal_alpha` lets
        // Mode 6 compete for mid-range alpha blocks. Bimodal blocks stay
        // excluded (worklog confirms Mode 6 for bimodal causes R/A regression
        // because the recompute-fair-error safety net forces bad selections).
        // FIX S (green max regression): skip Mode 6 for outline blocks where
        // green range > 32. Outline blocks (alpha_range <= 16) with high green
        // variance (e.g. deep-blue G=0 next to bright-cyan G=85) cause Mode 6's
        // 1-subset structure to drift green up to G=79 (was 56 in v3 before
        // Mode 6 expansion). Skipping Mode 6 ONLY for these blocks lets Mode 5
        // (Fix H) or Mode 7 (Fix N) handle them, both of which have outline-
        // specific green handling. For outline blocks with low green range
        // (<=32), Mode 6 is safe and its 16 selectors + 7-bit endpoints give
        // better R/B precision than Mode 5/7's 4 selectors — preserving the
        // R/B PSNR gains from Fix I (Mode 6 expansion).
        const bool outline_high_green = outline_alpha_block && (color_range_g > 48);
        if (pComp_params->m_alpha_settings.m_use_mode6 && !bimodal_alpha && !outline_high_green)
        {
                 color_cell_compressor_params params6 = *pParams;
                params6.m_weights[0] *= pComp_params->m_alpha_settings.m_mode67_error_weight_mul[0];
                params6.m_weights[1] *= pComp_params->m_alpha_settings.m_mode67_error_weight_mul[1];
                params6.m_weights[2] *= pComp_params->m_alpha_settings.m_mode67_error_weight_mul[2];
                params6.m_weights[3] *= pComp_params->m_alpha_settings.m_mode67_error_weight_mul[3];
                if (bimodal_alpha) params6.m_weights[1] *= 2; // FIX L: green upweight for bimodal
                params6.m_pSelector_weights = g_bc7_weights4;
                params6.m_pSelector_weightsx = (const vec4F *)&g_bc7_weights4x[0];
                params6.m_num_selector_weights = 16;
                params6.m_comp_bits = 7;
                params6.m_has_pbits = true;
                params6.m_endpoints_share_pbit = false;
                params6.m_has_alpha = true;
                                                
                int selectors[16];
                color_cell_compressor_results results6;
                results6.m_pSelectors = selectors;
                int selectors_temp6[16];
                results6.m_pSelectors_temp = selectors_temp6;
                uint64_t mode6_err = color_cell_compression(6, &params6, &results6, pComp_params, 16, pPixels, true);
                assert(mode6_err == results6.m_best_overall_err);
                
                // ---- RECOMPUTE FAIR ERROR ----
                if (bimodal_alpha)
                {
                        uint64_t true_err = 0;
                        color_quad_i low = results6.m_low_endpoint;
                        color_quad_i high = results6.m_high_endpoint;
                        uint32_t p0 = results6.m_pbits[0];
                        uint32_t p1 = results6.m_pbits[1];
                        color_quad_i scaled_low, scaled_high;
                        for (uint32_t c = 0; c < 4; c++)
                        {
                                uint32_t lv = ((low.m_c[c] << 1) | p0);
                                uint32_t hv = ((high.m_c[c] << 1) | p1);
                                lv = (lv << 1) | (lv >> 7);  // 7-bit + pbit → 8-bit
                                hv = (hv << 1) | (hv >> 7);
                                scaled_low.m_c[c] = lv;
                                scaled_high.m_c[c] = hv;
                        }
                        for (uint32_t i = 0; i < 16; i++)
                        {
                                uint32_t sel = selectors[i];
                                uint32_t w = g_bc7_weights4[sel];
                                color_quad_i decoded;
                                for (uint32_t c = 0; c < 4; c++)
                                        decoded.m_c[c] = (scaled_low.m_c[c] * (64 - w) + scaled_high.m_c[c] * w + 32) >> 6;
                                
                                int dr = decoded.m_c[0] - pPixels[i].m_c[0];
                                int dg = decoded.m_c[1] - pPixels[i].m_c[1];
                                int db = decoded.m_c[2] - pPixels[i].m_c[2];
                                int da = decoded.m_c[3] - pPixels[i].m_c[3];
                                true_err += pParams->m_weights[0] * (dr * dr);
                                true_err += pParams->m_weights[1] * (dg * dg);
                                true_err += pParams->m_weights[2] * (db * db);
                                true_err += pParams->m_weights[3] * (da * da);
                        }
                        mode6_err = true_err;
                }
                if (mode6_err < best_err)
                {
                        best_err = mode6_err;
                        opt_results.m_mode = 6;
                        opt_results.m_index_selector = 0;
                        opt_results.m_rotation = 0;
                        opt_results.m_partition = 0;
                        opt_results.m_low[0] = results6.m_low_endpoint;
                        opt_results.m_high[0] = results6.m_high_endpoint;
                        opt_results.m_pbits[0][0] = results6.m_pbits[0];
                        opt_results.m_pbits[0][1] = results6.m_pbits[1];
                        for ( int i = 0; i < 16; i++) opt_results.m_selectors[i] = selectors[i];
                }
        }

        // Mode 5
        if (pComp_params->m_alpha_settings.m_use_mode5)
        {
                // OPT-C (speed): max_color_range_m5 is the SAME value as the
                // max_color_range already computed above (from lo_r/hi_r/etc.
                // at the top of handle_alpha_block). The original code
                // recomputed it here with different variable names; remove
                // the duplicate 16-pixel scan.
                const int max_color_range_m5 = max_color_range;
                
                // FIX H (outline bug): evaluate Mode 5 for outline blocks even when
                // color range > 128. See outline_alpha_block definition above.
                // Mode 5's 7-bit RGB + 8-bit A endpoints have 4x the precision of
                // Mode 7's 5-bit and win for these blocks.
                if (outline_alpha_block || max_color_range_m5 < 128)
                {
                        color_cell_compressor_params params5 = *pParams;
                        const  int num_rotations = (pComp_params->m_perceptual || (!pComp_params->m_alpha_settings.m_use_mode5_rotation)) ? 2 : 4;
                        for ( uint32_t rotation = 0; rotation < num_rotations; rotation++)
                        {
                                        if ((pComp_params->m_mode5_rotation_mask & (1 << rotation)) == 0) continue;
                                        memcpy(params5.m_weights, pParams->m_weights, sizeof(params5.m_weights));
                                        if (bimodal_alpha) params5.m_weights[1] *= 2; // FIX L: green upweight for bimodal
                                        // FIX N5 (residual green outline, Mode 5): outline blocks (low alpha
                                        // variance) with a wide green range and one or two outlier high-green
                                        // pixels (e.g. a single G=43 pixel among 15 G=0 pixels) get their
                                        // green crushed because least-squares endpoint fit is dominated by
                                        // the majority G=0 pixels, pulling both green endpoints toward 0.
                                        // Upweighting green for these blocks makes the encoder prefer
                                        // endpoints that span the green range, preserving the outlier.
                                        // Graduated: 2x for G range > 32, 3x for G range > 64.
                                        if (outline_alpha_block) {
                                                if (color_range_g > 64) params5.m_weights[1] *= 3;
                                                else if (color_range_g > 32) params5.m_weights[1] *= 2;
                                        }
                                        if (rotation) swapu(&params5.m_weights[rotation - 1], &params5.m_weights[3]);

                                        color_quad_i rot_pixels[16];
                                        const  color_quad_i * pTrial_pixels = pPixels;
                                        uint32_t trial_lo_a = lo_a, trial_hi_a = hi_a;
                                        if (rotation)
                                        {
                                                        trial_lo_a = 255; trial_hi_a = 0;
                                                        for ( uint32_t i = 0; i < 16; i++)
                                                        {
                                                                        color_quad_i c = pPixels[i];
                                                                        swapi(&c.m_c[3], &c.m_c[rotation - 1]);
                                                                        rot_pixels[i] = c;
                                                                        trial_lo_a = minimumu(trial_lo_a, c.m_c[3]);
                                                                        trial_hi_a = maximumu(trial_hi_a, c.m_c[3]);
                                                        }
                                                        pTrial_pixels = rot_pixels;
                                        }

                                        bc7_optimization_results trial_opt_results5;
                                        uint64_t trial_mode5_err = 0;
                                        handle_alpha_block_mode5(pTrial_pixels, pComp_params, &params5, trial_lo_a, trial_hi_a, &trial_opt_results5, &trial_mode5_err);

                                        if (trial_mode5_err < best_err)
                                        {
                                                        best_err = trial_mode5_err;
                                                        opt_results = trial_opt_results5;
                                                        opt_results.m_rotation = rotation;
                                        }
                        } // rotation
                }
        }

        // Mode 7 (2-subset, 5-bit RGBA with pbits, 2-bit indices)
        const bool need_mode7_fallback = !near_opaque && !bimodal_alpha;
        // OPT-E (speed): Mode 7's 5-bit endpoints cannot compete with Mode 5's
        // 7-bit RGB + 8-bit alpha when Mode 5/6 already gave a near-perfect
        // fit. Mode 7's only advantage is its 2-subset structure (can split
        // dissimilar colors). For blocks where Mode 5/6 nailed it AND the
        // block isn't bimodal (needs alpha separation) or outline (needs
        // color separation), skip Mode 7's expensive partition search.
        // Threshold 32 = avg per-component squared err of 0.5 (RMSE ~0.7),
        // well below visual lossless. This saves ~24 est() calls + 8-16
        // color_cell_compression(7) calls per skipped block.
        const bool mode7_early_skip = (best_err < 32) && !bimodal_alpha && !outline_alpha_block;
        if (pComp_params->m_alpha_settings.m_use_mode7 && (!skip_mode7 || bimodal_alpha || need_mode7_fallback) && !mode7_early_skip)
        {
                solution solutions[BC7E_MAX_PARTITIONS7];
                uint32_t parts_to_try = pComp_params->m_alpha_settings.m_max_mode7_partitions_to_try;
                // FIX Q (residual green outline): outline blocks (low alpha variance,
                // high color variance) need more partition candidates to find one that
                // separates dissimilar colors (e.g. G=0 deep-blue from G=85 bright-cyan).
                // The default 2 partitions often misses the separating partition,
                // causing green to drift up to 73. Try 8 for outline blocks.
                if (outline_alpha_block && parts_to_try < 4) parts_to_try = 4;
                // FIX T (block-adaptive partition estimation): for bimodal alpha
                // blocks, use alpha-aware volume/compactness scoring so the
                // estimator prefers partitions that separate alpha=0 from
                // alpha=255 pixels. This reduces the worst-case alpha error
                // (was 161) by avoiding partitions that mix transparent and
                // opaque pixels in the same subset (which forces alpha
                // interpolation across the full 0-255 range). Outline blocks
                // (alpha_range <= 16) don't need this — alpha is nearly
                // constant so alpha-aware scoring would just add noise.
                const bool alpha_aware_part = bimodal_alpha && !outline_alpha_block;
                uint32_t num_solutions = estimate_partition_list(7, pPixels, pComp_params, solutions, parts_to_try, alpha_aware_part);
                color_cell_compressor_params params7 = *pParams;
                params7.m_weights[0] *= pComp_params->m_alpha_settings.m_mode67_error_weight_mul[0];
                params7.m_weights[1] *= pComp_params->m_alpha_settings.m_mode67_error_weight_mul[1];
                params7.m_weights[2] *= pComp_params->m_alpha_settings.m_mode67_error_weight_mul[2];
                params7.m_weights[3] *= pComp_params->m_alpha_settings.m_mode67_error_weight_mul[3];
                // FIX L (outline bug): for bimodal blocks, upweight green 2x in Mode 7.
                // Bimodal blocks have opaque deep-color pixels mixed with transparent.
                // Mode 7's partition estimator often groups opaque deep-purple (G=66)
                // with bright-cyan (G=255) in the same subset, causing green to drift
                // to G=139 (error 73). Upweighting green makes the estimator prefer
                // partitions that keep similar greens together, directly targeting
                // the "grayish green blockiness on deep purple" the user reported.
                if (bimodal_alpha) params7.m_weights[1] *= 2; // FIX L: green upweight for bimodal
                if (outline_alpha_block) params7.m_weights[1] *= 2; // FIX N: green upweight for outline
                params7.m_pSelector_weights = g_bc7_weights2;
                params7.m_pSelector_weightsx = (const vec4F *)&g_bc7_weights2x[0];
                params7.m_num_selector_weights = 4;
                params7.m_comp_bits = 5;
                params7.m_has_pbits = true;
                params7.m_endpoints_share_pbit = false;
                params7.m_has_alpha = true;
                int selectors_temp[16];
                // FIX R: when trying more than 2 partitions (Fix Q), force full
                // evaluation for outline/bimodal blocks. The fast evaluation path
                // (used when num_solutions > 2) picks the wrong partition for these
                // blocks, causing R/A regression. Full evaluation is slower but
                // correct.
                const  bool disable_faster_part_selection = outline_alpha_block; // FIX R: full eval for outline blocks
                for ( uint32_t solution_index = 0; solution_index < num_solutions; solution_index++)
                {
                        const uint32_t trial_partition = solutions[solution_index].m_index;
                        assert(trial_partition < 64);
                        const int *pPartition = &g_bc7_partition2[trial_partition * 16];
                        color_quad_i subset_colors[2][16];
                        uint32_t subset_total_colors7[2];
                        subset_total_colors7[0] = 0; subset_total_colors7[1] = 0;
                        int subset_pixel_index7[2][16];
                        int subset_selectors7[2][16];
                        color_cell_compressor_results subset_results7[2];
                        for ( uint32_t idx = 0; idx < 16; idx++)
                        {
                                const uint32_t p = pPartition[idx];
                                assert(p < 2);
                                subset_colors[p][subset_total_colors7[p]] = pPixels[idx];
                                subset_pixel_index7[p][subset_total_colors7[p]] = idx;
                                subset_total_colors7[p]++;
                        }
                        uint64_t trial_err = 0;
                        for ( uint32_t subset = 0; subset < 2; subset++)
                        {
                                color_cell_compressor_results * pResults = &subset_results7[subset];
                                pResults->m_pSelectors = &subset_selectors7[subset][0];
                                pResults->m_pSelectors_temp = selectors_temp;
                                uint64_t err = color_cell_compression(7, &params7, pResults, pComp_params, subset_total_colors7[subset], &subset_colors[subset][0], (num_solutions <= 2) || disable_faster_part_selection);
                                assert(err == pResults->m_best_overall_err);
                                trial_err += err;
                                if (trial_err > best_err) break;
                        } // subset
                        if (trial_err < best_err)
                        {
                                best_err = trial_err;
                                opt_results.m_mode = 7;
                                opt_results.m_index_selector = 0;
                                opt_results.m_rotation = 0;
                                opt_results.m_partition = trial_partition;
                                for ( uint32_t subset = 0; subset < 2; subset++)
                                {
                                        for ( uint32_t i = 0; i < subset_total_colors7[subset]; i++)
                                        {
                                                const uint32_t pixel_index = subset_pixel_index7[subset][i];
                                                opt_results.m_selectors[pixel_index] = subset_selectors7[subset][i];
                                        }
                                        opt_results.m_low[subset] = subset_results7[subset].m_low_endpoint;
                                        opt_results.m_high[subset] = subset_results7[subset].m_high_endpoint;
                                        opt_results.m_pbits[subset][0] = subset_results7[subset].m_pbits[0];
                                        opt_results.m_pbits[subset][1] = subset_results7[subset].m_pbits[1];
                                }
                        }
                } // solution_index
                if ((num_solutions > 2) && (opt_results.m_mode == 7) && (!disable_faster_part_selection))
                {
                        const uint32_t trial_partition = opt_results.m_partition;
                        assert(trial_partition < 64);
                        const int *pPartition = &g_bc7_partition2[trial_partition * 16];
                        color_quad_i subset_colors[2][16];
                        uint32_t subset_total_colors7[2];
                        subset_total_colors7[0] = 0; subset_total_colors7[1] = 0;
                        int subset_pixel_index7[2][16];
                        int subset_selectors7[2][16];
                        color_cell_compressor_results subset_results7[2];
                        for ( uint32_t idx = 0; idx < 16; idx++)
                        {
                                const uint32_t p = pPartition[idx];
                                assert(p < 2);
                                subset_colors[p][subset_total_colors7[p]] = pPixels[idx];
                                subset_pixel_index7[p][subset_total_colors7[p]] = idx;
                                subset_total_colors7[p]++;
                        }
                        uint64_t trial_err = 0;
                        for ( uint32_t subset = 0; subset < 2; subset++)
                        {
                                color_cell_compressor_results * pResults = &subset_results7[subset];
                                pResults->m_pSelectors = &subset_selectors7[subset][0];
                                pResults->m_pSelectors_temp = selectors_temp;
                                uint64_t err = color_cell_compression(7, &params7, pResults, pComp_params, subset_total_colors7[subset], &subset_colors[subset][0], true);
                                assert(err == pResults->m_best_overall_err);
                                trial_err += err;
                                if (trial_err > best_err) break;
                        } // subset
                        if (trial_err < best_err)
                        {
                                best_err = trial_err;
                                for ( uint32_t subset = 0; subset < 2; subset++)
                                {
                                        for ( uint32_t i = 0; i < subset_total_colors7[subset]; i++)
                                        {
                                                const uint32_t pixel_index = subset_pixel_index7[subset][i];
                                                opt_results.m_selectors[pixel_index] = subset_selectors7[subset][i];
                                        }
                                        opt_results.m_low[subset] = subset_results7[subset].m_low_endpoint;
                                        opt_results.m_high[subset] = subset_results7[subset].m_high_endpoint;
                                        opt_results.m_pbits[subset][0] = subset_results7[subset].m_pbits[0];
                                        opt_results.m_pbits[subset][1] = subset_results7[subset].m_pbits[1];
                                }
                        }
                }
        }

        // FIX FM (Force Mode 5/6 for bias-prone Mode 4 alpha blocks):
        // Mode 4's 5-bit RGB endpoint grid (step 8) can cause systematic green
        // DC bias when the optimal endpoint falls between grid points, visible
        // as "patched brightening" / "blocky darkening" on the green channel.
        // If Mode 4 won AND the green DC bias exceeds 1.5, try forcing Mode 6
        // (7-bit RGB + 7-bit A + pbit = 8-bit, 16 shared selectors) first — it
        // preserves alpha precision better than Mode 5. If Mode 6 doesn't help
        // (or isn't available), fall back to Mode 5 (7-bit RGB + 8-bit A, 4
        // selectors). Accept the forced mode only if:
        //   (1) its green DC bias is actually lower than Mode 4's, AND
        //   (2) its green SSE is not more than 1.5x Mode 4's (prevents the
        //       forced mode from creating worse max-error artifacts, e.g.
        //       Mode 6's 1-subset LSQ crushing outlier pixels).
        // Skip if rotation != 0 (rare case; green lives in a different slot).
        if (opt_results.m_mode == 4 && opt_results.m_rotation == 0)
        {
                // Decode winning Mode 4 green DC (5-bit RGB, no pbit, n=5)
                const uint32_t n5 = 5;
                uint32_t aLowG  = (opt_results.m_low[0].m_c[1]  << (8 - n5)) | (opt_results.m_low[0].m_c[1]  >> (2*n5 - 8));
                uint32_t aHighG = (opt_results.m_high[0].m_c[1] << (8 - n5)) | (opt_results.m_high[0].m_c[1] >> (2*n5 - 8));
                const uint32_t *sw_m4 = (opt_results.m_index_selector == 0) ? g_bc7_weights2 : g_bc7_weights3;
                int m4_dec_g = 0, m4_in_g = 0;
                uint64_t m4_sse_g = 0;
                for (uint32_t i = 0; i < 16; i++) {
                        uint32_t sel = opt_results.m_selectors[i];
                        uint32_t w = sw_m4[sel];
                        int dg = (int)(((64 - w) * aLowG + w * aHighG + 32) >> 6);
                        m4_dec_g += dg;
                        m4_in_g  += pPixels[i].m_c[1];
                        int e = dg - pPixels[i].m_c[1];
                        m4_sse_g += (uint64_t)(e * e);
                }
                float m4_bias = (float)(m4_dec_g - m4_in_g) / 16.0f;

                if (m4_bias > 1.5f || m4_bias < -1.5f) {
                        bool replaced = false;

                        // Try Mode 6 first (better alpha precision than Mode 5).
                        // Skip for outline blocks (alpha_range <= 16): Mode 6's shared
                        // color+alpha selectors can't represent "constant alpha, varying
                        // color" well, causing alpha regression. Outline blocks fall
                        // through to Mode 5 (separate color/alpha selectors).
                        if (!replaced && pComp_params->m_alpha_settings.m_use_mode6 &&
                            !bimodal_alpha && !outline_alpha_block)
                        {
                                color_cell_compressor_params params6 = *pParams;
                                params6.m_weights[0] *= pComp_params->m_alpha_settings.m_mode67_error_weight_mul[0];
                                params6.m_weights[1] *= pComp_params->m_alpha_settings.m_mode67_error_weight_mul[1];
                                params6.m_weights[2] *= pComp_params->m_alpha_settings.m_mode67_error_weight_mul[2];
                                params6.m_weights[3] *= pComp_params->m_alpha_settings.m_mode67_error_weight_mul[3];
                                params6.m_pSelector_weights = g_bc7_weights4;
                                params6.m_pSelector_weightsx = (const vec4F *)&g_bc7_weights4x[0];
                                params6.m_num_selector_weights = 16;
                                params6.m_comp_bits = 7;
                                params6.m_has_pbits = true;
                                params6.m_endpoints_share_pbit = false;
                                params6.m_has_alpha = true;
                                params6.m_perceptual = pComp_params->m_perceptual;

                                int selectors6[16];
                                color_cell_compressor_results results6;
                                results6.m_pSelectors = selectors6;
                                int selectors_temp6[16];
                                results6.m_pSelectors_temp = selectors_temp6;
                                color_cell_compression(6, &params6, &results6, pComp_params, 16, pPixels, true);

                                // Compute Mode 6 green DC bias AND SSE (7-bit + pbit = 8-bit)
                                int m6_dec_g = 0;
                                uint64_t m6_sse_g = 0;
                                for (uint32_t i = 0; i < 16; i++) {
                                        uint32_t sel = selectors6[i];
                                        uint32_t w = g_bc7_weights4[sel];
                                        uint32_t p0 = results6.m_pbits[0];
                                        uint32_t p1 = results6.m_pbits[1];
                                        uint32_t aLowG6  = (results6.m_low_endpoint.m_c[1]  << 1) | p0;
                                        uint32_t aHighG6 = (results6.m_high_endpoint.m_c[1] << 1) | p1;
                                        int dg = (int)(((64 - w) * aLowG6 + w * aHighG6 + 32) >> 6);
                                        m6_dec_g += dg;
                                        int e = dg - pPixels[i].m_c[1];
                                        m6_sse_g += (uint64_t)(e * e);
                                }
                                float m6_bias = (float)(m6_dec_g - m4_in_g) / 16.0f;

                                // Accept Mode 6 only if: (1) reduces |bias|, AND (2) SSE <= 1.5x Mode 4
                                if (fabsf(m6_bias) < fabsf(m4_bias) && m6_sse_g <= (m4_sse_g * 3 + 1) / 2) {
                                        opt_results.m_mode = 6;
                                        opt_results.m_index_selector = 0;
                                        opt_results.m_rotation = 0;
                                        opt_results.m_partition = 0;
                                        opt_results.m_low[0] = results6.m_low_endpoint;
                                        opt_results.m_high[0] = results6.m_high_endpoint;
                                        opt_results.m_pbits[0][0] = results6.m_pbits[0];
                                        opt_results.m_pbits[0][1] = results6.m_pbits[1];
                                        for (uint32_t i = 0; i < 16; i++) opt_results.m_selectors[i] = selectors6[i];
                                        replaced = true;
                                }
                        }

                        // Fall back to Mode 5 if Mode 6 didn't help
                        if (!replaced && pComp_params->m_alpha_settings.m_use_mode5)
                        {
                                color_cell_compressor_params params5 = *pParams;
                                bc7_optimization_results trial_opt_results5;
                                uint64_t trial_mode5_err = 0;
                                handle_alpha_block_mode5(pPixels, pComp_params, &params5, lo_a, hi_a, &trial_opt_results5, &trial_mode5_err);

                                // Compute Mode 5 green DC bias AND SSE (7-bit RGB, no pbit, n=7)
                                const uint32_t n7 = 7;
                                uint32_t m5_aLowG  = (trial_opt_results5.m_low[0].m_c[1]  << (8 - n7)) | (trial_opt_results5.m_low[0].m_c[1]  >> (2*n7 - 8));
                                uint32_t m5_aHighG = (trial_opt_results5.m_high[0].m_c[1] << (8 - n7)) | (trial_opt_results5.m_high[0].m_c[1] >> (2*n7 - 8));
                                int m5_dec_g = 0;
                                uint64_t m5_sse_g = 0;
                                for (uint32_t i = 0; i < 16; i++) {
                                        uint32_t sel = trial_opt_results5.m_selectors[i];
                                        uint32_t w = g_bc7_weights2[sel];
                                        int dg = (int)(((64 - w) * m5_aLowG + w * m5_aHighG + 32) >> 6);
                                        m5_dec_g += dg;
                                        int e = dg - pPixels[i].m_c[1];
                                        m5_sse_g += (uint64_t)(e * e);
                                }
                                float m5_bias = (float)(m5_dec_g - m4_in_g) / 16.0f;

                                // Accept Mode 5 only if: (1) reduces |bias|, AND (2) SSE <= 1.5x Mode 4
                                if (fabsf(m5_bias) < fabsf(m4_bias) && m5_sse_g <= (m4_sse_g * 3 + 1) / 2) {
                                        opt_results = trial_opt_results5;
                                        opt_results.m_rotation = 0;
                                }
                        }
                }
        }

        encode_bc7_block(pBlock, &opt_results);
}

static void handle_opaque_block(void * pBlock, const  color_quad_i * pPixels, const  bc7e_compress_block_params * pComp_params,  color_cell_compressor_params * pParams, const bc7_block_stats * pStats)
{
                int selectors_temp[16];
                                
                bc7_optimization_results opt_results;
                                
                uint64_t best_err = UINT64_MAX;
                uint64_t best_non_3subset_err = UINT64_MAX;  // Track best non-3-subset error

                // QuickBC7 mode pruning: skip multi-subset modes for simple blocks.
                // Luma range and distinct color count determine which modes are worth trying.
                const int luma_range = pStats->max_luma - pStats->min_luma;
                const int ndc = pStats->num_distinct_colors;

                // Existing logic for skipping multi-subset entirely
                const bool skip_multisubset = (ndc <= 3) || (luma_range <= 8 && pStats->luma_var < 64);

                // NEW: Only try heavy 3-subset modes (0 and 2) if the block actually has complex details.
                // High distinct color count or high variance indicates fine details/edges.
                const bool try_3_subset = (ndc >= 6) || (pStats->luma_var > 250);

                // NOTE: Fix I (block-adaptive alpha boost) is intentionally NOT
                // applied to opaque blocks. Opaque blocks have alpha=255 (range=0),
                // so dom_c can only be R/G/B. Testing showed RGB boosts in opaque
                // blocks cause B regression (max B 53->110) by shifting mode
                // selection (e.g. Mode 5 winning over Mode 6). Mode 6 (1-subset,
                // 8-bit endpoints, 16 selectors) is already optimal for most opaque
                // blocks, and the existing partition estimator picks the right mode.

                // Mode 6
                if (pComp_params->m_opaque_settings.m_use_mode[6])
                {
                                pParams->m_pSelector_weights = g_bc7_weights4;
                                pParams->m_pSelector_weightsx = (const vec4F * )&g_bc7_weights4x[0];
                                pParams->m_num_selector_weights = 16;

                                pParams->m_comp_bits = 7;
                                pParams->m_has_pbits = true;
                                pParams->m_endpoints_share_pbit = false;

                                pParams->m_perceptual = pComp_params->m_perceptual;
                                                                
                                color_cell_compressor_results results6;                                         
                                results6.m_pSelectors = opt_results.m_selectors;
                                results6.m_pSelectors_temp = selectors_temp;

                                best_err = color_cell_compression(6, pParams, &results6, pComp_params, 16, pPixels, true);
                                best_non_3subset_err = best_err;
                                                                                                
                                opt_results.m_mode = 6;
                                opt_results.m_index_selector = 0;
                                opt_results.m_rotation = 0;
                                opt_results.m_partition = 0;

                                opt_results.m_low[0] = results6.m_low_endpoint;
                                opt_results.m_high[0] = results6.m_high_endpoint;

                                opt_results.m_pbits[0][0] = results6.m_pbits[0];
                                opt_results.m_pbits[0][1] = results6.m_pbits[1];
                }

        const uint32_t MAX_MODE13_ADAPTIVE = 8; 
        solution solutions2[MAX_MODE13_ADAPTIVE];
        uint32_t num_solutions2 = 0;
        
        if (pComp_params->m_opaque_settings.m_use_mode[1] || pComp_params->m_opaque_settings.m_use_mode[3])
        {
                // ==========================================
                // THE "MODE STEALING" PREVENTION TRICK
                // If there are more than 3 colors, a 2-subset mode CANNOT 
                // win without looking blocky. Intentionally cripple it 
                // so the 3-subset modes (0 and 2) are forced to handle it.
                // ==========================================
                uint32_t dynamic_parts_13 = 8; // Default for normal images

                if (pStats->num_distinct_colors > 3) {
                        // Force 2-subset modes to fail fast.
                        // Just check the absolute basic shapes to prove it's a bad fit.
                        dynamic_parts_13 = 3; 
                }
                else if (pStats->luma_var < 1000) {
                        dynamic_parts_13 = 4; 
                }
                
                if (pComp_params->m_opaque_settings.m_max_mode13_partitions_to_try == 1)
                {
                        solutions2[0].m_index = quick_estimate_partition_2subset(pPixels, pStats, pComp_params);
                        num_solutions2 = 1;
                }
                else
                {
                        num_solutions2 = estimate_partition_list(1, pPixels, pComp_params, solutions2, dynamic_parts_13);
                }
                // ==========================================
        }
                                
                const  bool disable_faster_part_selection = false;
                                                                                                                                
                // Mode 1 (2-subset, 6-bit color, 3-bit indices)
                if (pComp_params->m_opaque_settings.m_use_mode[1] && !skip_multisubset)
                {
                                pParams->m_pSelector_weights = g_bc7_weights3;
                                pParams->m_pSelector_weightsx = (const vec4F *)&g_bc7_weights3x[0];
                                pParams->m_num_selector_weights = 8;

                                pParams->m_comp_bits = 6;
                                pParams->m_has_pbits = true;
                                pParams->m_endpoints_share_pbit = true;

                                pParams->m_perceptual = pComp_params->m_perceptual;

                                for ( uint32_t solution_index = 0; solution_index < num_solutions2; solution_index++)
                                {
                                                const uint32_t trial_partition = solutions2[solution_index].m_index;
                                                assert(trial_partition < 64);

                                                const int *pPartition = &g_bc7_partition2[trial_partition * 16];
                                                                                                
                                                color_quad_i subset_colors[2][16];

                                                uint32_t subset_total_colors1[2];
                                                subset_total_colors1[0] = 0;
                                                subset_total_colors1[1] = 0;
                                                                
                                                int subset_pixel_index1[2][16];
                                                int subset_selectors1[2][16];
                                                color_cell_compressor_results subset_results1[2];

                                                for ( uint32_t idx = 0; idx < 16; idx++)
                                                {
// #pragma ignore warning(perf)  [ISPC pragma, no-op in C++]
                                                                const uint32_t p = pPartition[idx];
                                                                assert(p < 2);

// #pragma ignore warning(perf)  [ISPC pragma, no-op in C++]
                                                                subset_colors[p][subset_total_colors1[p]] = pPixels[idx];
// #pragma ignore warning(perf)  [ISPC pragma, no-op in C++]
                                                                subset_pixel_index1[p][subset_total_colors1[p]] = idx;
// #pragma ignore warning(perf)  [ISPC pragma, no-op in C++]
                                                                subset_total_colors1[p]++;
                                                }
                                                                                                                                
                                                uint64_t trial_err = 0;
                                                for ( uint32_t subset = 0; subset < 2; subset++)
                                                {
                                                                 color_cell_compressor_results * pResults = &subset_results1[subset];

                                                                pResults->m_pSelectors = &subset_selectors1[subset][0];
                                                                pResults->m_pSelectors_temp = selectors_temp;

                                                                uint64_t err = color_cell_compression(1, pParams, pResults, pComp_params, subset_total_colors1[subset], &subset_colors[subset][0], (num_solutions2 <= 2) || disable_faster_part_selection);
                                                                assert(err == pResults->m_best_overall_err);

                                                                trial_err += err;
                                                                if (trial_err > best_err)
                                                                                break;
                                                                                
                                                } // subset

                                                if (trial_err < best_err)
                                                {
                                                                best_err = trial_err;
                                                        best_non_3subset_err = trial_err;

                                                                opt_results.m_mode = 1;
                                                                opt_results.m_index_selector = 0;
                                                                opt_results.m_rotation = 0;
                                                                opt_results.m_partition = trial_partition;

                                                                for ( uint32_t subset = 0; subset < 2; subset++)
                                                                {
                                                                                for ( uint32_t i = 0; i < subset_total_colors1[subset]; i++)
                                                                                {
                                                                                                const uint32_t pixel_index = subset_pixel_index1[subset][i];

// #pragma ignore warning(perf)  [ISPC pragma, no-op in C++]
                                                                                                opt_results.m_selectors[pixel_index] = subset_selectors1[subset][i];
                                                                                }

                                                                                opt_results.m_low[subset] = subset_results1[subset].m_low_endpoint;
                                                                                opt_results.m_high[subset] = subset_results1[subset].m_high_endpoint;

                                                                                opt_results.m_pbits[subset][0] = subset_results1[subset].m_pbits[0];
                                                                }
                                                }
                                }

                                if ((num_solutions2 > 2) && (opt_results.m_mode == 1) && (!disable_faster_part_selection))
                                {
                                                const uint32_t trial_partition = opt_results.m_partition;
                                                assert(trial_partition < 64);

                                                const int *pPartition = &g_bc7_partition2[trial_partition * 16];
                                                                                                
                                                color_quad_i subset_colors[2][16];

                                                uint32_t subset_total_colors1[2];
                                                subset_total_colors1[0] = 0;
                                                subset_total_colors1[1] = 0;
                                                                
                                                int subset_pixel_index1[2][16];
                                                int subset_selectors1[2][16];
                                                color_cell_compressor_results subset_results1[2];

                                                for ( uint32_t idx = 0; idx < 16; idx++)
                                                {
// #pragma ignore warning(perf)  [ISPC pragma, no-op in C++]
                                                                const uint32_t p = pPartition[idx];
                                                                assert(p < 2);

// #pragma ignore warning(perf)  [ISPC pragma, no-op in C++]
                                                                subset_colors[p][subset_total_colors1[p]] = pPixels[idx];
// #pragma ignore warning(perf)  [ISPC pragma, no-op in C++]
                                                                subset_pixel_index1[p][subset_total_colors1[p]] = idx;
// #pragma ignore warning(perf)  [ISPC pragma, no-op in C++]
                                                                subset_total_colors1[p]++;
                                                }
                                                                                                                                
                                                uint64_t trial_err = 0;
                                                for ( uint32_t subset = 0; subset < 2; subset++)
                                                {
                                                                 color_cell_compressor_results * pResults = &subset_results1[subset];

                                                                pResults->m_pSelectors = &subset_selectors1[subset][0];
                                                                pResults->m_pSelectors_temp = selectors_temp;

                                                                uint64_t err = color_cell_compression(1, pParams, pResults, pComp_params, subset_total_colors1[subset], &subset_colors[subset][0], true);
                                                                assert(err == pResults->m_best_overall_err);

                                                                trial_err += err;
                                                                if (trial_err > best_err)
                                                                                break;
                                                                                
                                                } // subset

                                                if (trial_err < best_err)
                                                {
                                                                best_err = trial_err;
                                                        best_non_3subset_err = trial_err;

                                                                for ( uint32_t subset = 0; subset < 2; subset++)
                                                                {
                                                                                for ( uint32_t i = 0; i < subset_total_colors1[subset]; i++)
                                                                                {
                                                                                                const uint32_t pixel_index = subset_pixel_index1[subset][i];
// #pragma ignore warning(perf)  [ISPC pragma, no-op in C++]
                                                                                                opt_results.m_selectors[pixel_index] = subset_selectors1[subset][i];
                                                                                }

                                                                                opt_results.m_low[subset] = subset_results1[subset].m_low_endpoint;
                                                                                opt_results.m_high[subset] = subset_results1[subset].m_high_endpoint;

                                                                                opt_results.m_pbits[subset][0] = subset_results1[subset].m_pbits[0];
                                                                }
                                                }
                                }
                }
                                
                // Mode 0 (3-subset, 4-bit color, 3-bit indices)
        // OPT-B (speed): Mode 0 is a 3-subset 4-bit-color mode that almost
        // never wins in practice (0% of blocks on sheet.png, typically <1%
        // on natural images — Mode 2's 5-bit endpoints and Mode 1's 2-subset
        // 6-bit endpoints both dominate it). When Mode 6 + Mode 1 have already
        // fit the block well (low best_err), Mode 0 cannot catch up — its
        // 4-bit endpoint precision caps how good it can be. Skipping it in
        // that case saves the expensive estimate_partition(0) call (192 est()
        // calls per try_3_subset block). When best_err is HIGH (Mode 6+1
        // failed to fit), we still try Mode 0 because its 3-subset structure
        // + 3-bit selectors (8 levels) can occasionally win on blocks with
        // 3 distinct color clusters. Threshold 256 = avg per-component squared
        // err of 4 (RMSE 2.0), well above the visual-lossless threshold.
        const bool try_mode0 = try_3_subset && (best_err > 256);
        if (pComp_params->m_opaque_settings.m_use_mode[0] && !skip_multisubset && try_mode0)
        {
                solution solutions3[BC7E_MAX_PARTITIONS0];
                uint32_t num_solutions3 = 0;
                
                if (pComp_params->m_opaque_settings.m_max_mode0_partitions_to_try == 1)
                {
                        solutions3[0].m_index = estimate_partition(0, pPixels, pComp_params);
                        num_solutions3 = 1;
                }
                else
                {
                                                // Adaptive Mode-0 partition logic
                                                // (this is the secret sauce to 100% PNSR)
                                                uint32_t dynamic_parts_0 = 8;

                                                if (pStats->luma_var < 500) dynamic_parts_0 = 4;
                                                else if (pStats->luma_var > 10000) dynamic_parts_0 = 16;

                                                part_score top_parts0[16]; // Use a fixed max array size of 16
                                                find_top_mode2_partitions(pPixels, top_parts0, dynamic_parts_0);
                        
                        num_solutions3 = 0;
                        for (uint32_t i = 0; i < pComp_params->m_opaque_settings.m_max_mode0_partitions_to_try; i++)
                        {
                                if (top_parts0[i].score == UINT64_MAX) continue;
                                
                                solutions3[num_solutions3].m_index = top_parts0[i].index;
                                num_solutions3++;
                        }
                        // ==========================================
                }

                pParams->m_pSelector_weights = g_bc7_weights3;
                pParams->m_pSelector_weightsx = (const vec4F *)&g_bc7_weights3x[0];
                pParams->m_num_selector_weights = 8;

                pParams->m_comp_bits = 4;
                pParams->m_has_pbits = true;
                pParams->m_endpoints_share_pbit = false;

                                pParams->m_perceptual = pComp_params->m_perceptual;

                                                                // ============================================================
                                                                // TWO-STAGE PARTITION EVALUATION (Optimization #3)
                                                                // Stage 1: Quick-scan all candidate partitions with
                                                                //          refinement=false (1 find_optimal_solution per
                                                                //          subset). Track the partition with lowest error.
                                                                // Stage 2: Run full refinement=true only on the winning
                                                                //          partition from stage 1.
                                                                // ============================================================
                                                                uint32_t best_part_idx = 0;
                                                                uint64_t best_stage1_err = UINT64_MAX;

                                                                // ---- Stage 1: quick scan ----
                                                                for ( uint32_t solution_index = 0; solution_index < num_solutions3; solution_index++)
                                                                {
                                                                                const uint32_t best_partition0 = solutions3[solution_index].m_index;
                                                                                const int *pPartition = &g_bc7_partition3[best_partition0 * 16];

                                                                                color_quad_i subset_colors_s1[3][16];
                                                                                uint32_t subset_total_s1[3] = { 0, 0, 0 };

                                                                                for ( uint32_t idx = 0; idx < 16; idx++)
                                                                                {
                                                                                                const uint32_t p = pPartition[idx];
                                                                                                subset_colors_s1[p][subset_total_s1[p]] = pPixels[idx];
                                                                                                subset_total_s1[p]++;
                                                                                }

                                                                                color_cell_compressor_results stage1_results[3];
                                                                                int stage1_selectors[3][16];

                                                                                uint64_t part_err = 0;
                                                                                for ( uint32_t subset = 0; subset < 3; subset++)
                                                                                {
                                                                                                color_cell_compressor_results * pResults = &stage1_results[subset];
                                                                                                pResults->m_pSelectors = &stage1_selectors[subset][0];
                                                                                                pResults->m_pSelectors_temp = selectors_temp;

                                                                                                uint64_t err = color_cell_compression(0, pParams, pResults, pComp_params, subset_total_s1[subset], &subset_colors_s1[subset][0], false);

                                                                                                part_err += err;
                                                                                                if (part_err > best_stage1_err)
                                                                                                                break;
                                                                                } // subset

                                                                                if (part_err < best_stage1_err)
                                                                                {
                                                                                                best_stage1_err = part_err;
                                                                                                best_part_idx = solution_index;
                                                                                }
                                                                } // stage 1

                                                                // ---- Stage 2: full refinement on the winner only ----
                                                                if (num_solutions3 > 0)
                                                                {
                                                                                const uint32_t best_partition0 = solutions3[best_part_idx].m_index;
                                                                                const int *pPartition = &g_bc7_partition3[best_partition0 * 16];

                                                                                color_quad_i subset_colors[3][16];
                                                                                uint32_t subset_total_colors0[3];
                                                                                subset_total_colors0[0] = 0;
                                                                                subset_total_colors0[1] = 0;
                                                                                subset_total_colors0[2] = 0;

                                                                                int subset_pixel_index0[3][16];

                                                                                for ( uint32_t idx = 0; idx < 16; idx++)
                                                                                {
                                                                                                const uint32_t p = pPartition[idx];
                                                                                                subset_colors[p][subset_total_colors0[p]] = pPixels[idx];
                                                                                                subset_pixel_index0[p][subset_total_colors0[p]] = idx;
                                                                                                subset_total_colors0[p]++;
                                                                                }

                                                                                color_cell_compressor_results subset_results0[3];
                                                                                int subset_selectors0[3][16];

                                                                                uint64_t mode0_err = 0;
                                                                                for ( uint32_t subset = 0; subset < 3; subset++)
                                                                                {
                                                                                                 color_cell_compressor_results * pResults = &subset_results0[subset];

                                                                                                pResults->m_pSelectors = &subset_selectors0[subset][0];
                                                                                                pResults->m_pSelectors_temp = selectors_temp;

                                                                                                uint64_t err = color_cell_compression(0, pParams, pResults, pComp_params, subset_total_colors0[subset], &subset_colors[subset][0], true);
                                                                                                assert(err == pResults->m_best_overall_err);

                                                                                                mode0_err += err;
                                                                                } // subset

                                                                                if (mode0_err < best_err)
                                                                                {
                                                                                                best_err = mode0_err;

                                                                                                opt_results.m_mode = 0;
                                                                                                opt_results.m_index_selector = 0;
                                                                                                opt_results.m_rotation = 0;
                                                                                                opt_results.m_partition = best_partition0;

                                                                                                for ( uint32_t subset = 0; subset < 3; subset++)
                                                                                                {
                                                                                                                for ( uint32_t i = 0; i < subset_total_colors0[subset]; i++)
                                                                                                                {
                                                                                                                                const uint32_t pixel_index = subset_pixel_index0[subset][i];
                                                                                                                                opt_results.m_selectors[pixel_index] = subset_selectors0[subset][i];
                                                                                                                }

                                                                                                        opt_results.m_low[subset] = subset_results0[subset].m_low_endpoint;
                                                                                                        opt_results.m_high[subset] = subset_results0[subset].m_high_endpoint;

                                                                                                        opt_results.m_pbits[subset][0] = subset_results0[subset].m_pbits[0];
                                                                                                        opt_results.m_pbits[subset][1] = subset_results0[subset].m_pbits[1];
                                                                                                }
                                                                                }
                                                                } // stage 2
                                                }
                                
                // Mode 3 (2-subset, 7+1-bit color with shared pbit, 2-bit indices)
                if (pComp_params->m_opaque_settings.m_use_mode[3] && !skip_multisubset && try_3_subset)
                {
                                pParams->m_pSelector_weights = g_bc7_weights2;
                                pParams->m_pSelector_weightsx = (const vec4F *)&g_bc7_weights2x[0];
                                pParams->m_num_selector_weights = 4;

                                pParams->m_comp_bits = 7;
                                pParams->m_has_pbits = true;
                                pParams->m_endpoints_share_pbit = false;

                                pParams->m_perceptual = pComp_params->m_perceptual;

                                for ( uint32_t solution_index = 0; solution_index < num_solutions2; solution_index++)
                                {
                                                const uint32_t trial_partition = solutions2[solution_index].m_index;
                                                assert(trial_partition < 64);

                                                const int *pPartition = &g_bc7_partition2[trial_partition * 16];

                                                color_quad_i subset_colors[2][16];

                                                uint32_t subset_total_colors3[2];
                                                subset_total_colors3[0] = 0;
                                                subset_total_colors3[1] = 0;
                                                 
                                                int subset_pixel_index3[2][16];
                                                int subset_selectors3[2][16];
                                                color_cell_compressor_results subset_results3[2];

                                                for ( uint32_t idx = 0; idx < 16; idx++)
                                                {
// #pragma ignore warning(perf)  [ISPC pragma, no-op in C++]
                                                                const uint32_t p = pPartition[idx];
                                                                assert(p < 2);

// #pragma ignore warning(perf)  [ISPC pragma, no-op in C++]
                                                                subset_colors[p][subset_total_colors3[p]] = pPixels[idx];

// #pragma ignore warning(perf)  [ISPC pragma, no-op in C++]
                                                                subset_pixel_index3[p][subset_total_colors3[p]] = idx;

// #pragma ignore warning(perf)  [ISPC pragma, no-op in C++]
                                                                subset_total_colors3[p]++;
                                                }

                                                uint64_t trial_err = 0;
                                                for ( uint32_t subset = 0; subset < 2; subset++)
                                                {
                                                                 color_cell_compressor_results * pResults = &subset_results3[subset];

                                                                pResults->m_pSelectors = &subset_selectors3[subset][0];
                                                                pResults->m_pSelectors_temp = selectors_temp;

                                                                uint64_t err = color_cell_compression(3, pParams, pResults, pComp_params, subset_total_colors3[subset], &subset_colors[subset][0], (num_solutions2 <= 2) || disable_faster_part_selection);
                                                                assert(err == pResults->m_best_overall_err);

                                                                trial_err += err;
                                                                if (trial_err > best_err)
                                                                                break;
                                                } // subset

                                                if (trial_err < best_err)
                                                {
                                                                best_err = trial_err;
                                                        best_non_3subset_err = trial_err;
                                                                                                                                                                
                                                                opt_results.m_mode = 3;
                                                                opt_results.m_index_selector = 0;
                                                                opt_results.m_rotation = 0;
                                                                opt_results.m_partition = trial_partition;

                                                                for ( uint32_t subset = 0; subset < 2; subset++)
                                                                {
                                                                                for ( uint32_t i = 0; i < subset_total_colors3[subset]; i++)
                                                                                {
                                                                                                const uint32_t pixel_index = subset_pixel_index3[subset][i];

// #pragma ignore warning(perf)  [ISPC pragma, no-op in C++]
                                                                                                opt_results.m_selectors[pixel_index] = subset_selectors3[subset][i];
                                                                                }

                                                                                opt_results.m_low[subset] = subset_results3[subset].m_low_endpoint;
                                                                                opt_results.m_high[subset] = subset_results3[subset].m_high_endpoint;

                                                                                opt_results.m_pbits[subset][0] = subset_results3[subset].m_pbits[0];
                                                                                opt_results.m_pbits[subset][1] = subset_results3[subset].m_pbits[1];
                                                                }
                                                }

                                } // solution_index

                                if ((num_solutions2 > 2) && (opt_results.m_mode == 3) && (!disable_faster_part_selection))
                                {
                                                const uint32_t trial_partition = opt_results.m_partition;
                                                assert(trial_partition < 64);

                                                const int *pPartition = &g_bc7_partition2[trial_partition * 16];

                                                color_quad_i subset_colors[2][16];

                                                uint32_t subset_total_colors3[2];
                                                subset_total_colors3[0] = 0;
                                                subset_total_colors3[1] = 0;
                                                 
                                                int subset_pixel_index3[2][16];
                                                int subset_selectors3[2][16];
                                                color_cell_compressor_results subset_results3[2];

                                                for ( uint32_t idx = 0; idx < 16; idx++)
                                                {
// #pragma ignore warning(perf)  [ISPC pragma, no-op in C++]
                                                                const uint32_t p = pPartition[idx];
                                                                assert(p < 2);

// #pragma ignore warning(perf)  [ISPC pragma, no-op in C++]
                                                                subset_colors[p][subset_total_colors3[p]] = pPixels[idx];

// #pragma ignore warning(perf)  [ISPC pragma, no-op in C++]
                                                                subset_pixel_index3[p][subset_total_colors3[p]] = idx;

// #pragma ignore warning(perf)  [ISPC pragma, no-op in C++]
                                                                subset_total_colors3[p]++;
                                                }

                                                uint64_t trial_err = 0;
                                                for ( uint32_t subset = 0; subset < 2; subset++)
                                                {
                                                                 color_cell_compressor_results * pResults = &subset_results3[subset];

                                                                pResults->m_pSelectors = &subset_selectors3[subset][0];
                                                                pResults->m_pSelectors_temp = selectors_temp;

                                                                uint64_t err = color_cell_compression(3, pParams, pResults, pComp_params, subset_total_colors3[subset], &subset_colors[subset][0], true);
                                                                assert(err == pResults->m_best_overall_err);

                                                                trial_err += err;
                                                                if (trial_err > best_err)
                                                                                break;
                                                } // subset

                                                if (trial_err < best_err)
                                                {
                                                                best_err = trial_err;
                                                        best_non_3subset_err = trial_err;
                                                                                                                                                                
                                                                for ( uint32_t subset = 0; subset < 2; subset++)
                                                                {
                                                                                for ( uint32_t i = 0; i < subset_total_colors3[subset]; i++)
                                                                                {
                                                                                                const uint32_t pixel_index = subset_pixel_index3[subset][i];

// #pragma ignore warning(perf)  [ISPC pragma, no-op in C++]
                                                                                                opt_results.m_selectors[pixel_index] = subset_selectors3[subset][i];
                                                                                }

                                                                                opt_results.m_low[subset] = subset_results3[subset].m_low_endpoint;
                                                                                opt_results.m_high[subset] = subset_results3[subset].m_high_endpoint;

                                                                                opt_results.m_pbits[subset][0] = subset_results3[subset].m_pbits[0];
                                                                                opt_results.m_pbits[subset][1] = subset_results3[subset].m_pbits[1];
                                                                }
                                                }
                                }
                }

                // Mode 5 (1-subset, 7-bit color + 8-bit alpha, separate selectors)
                // Only useful for opaque blocks when using rotation to treat one RGB channel as alpha.
                if ((!pComp_params->m_perceptual) && (pComp_params->m_opaque_settings.m_use_mode[5]) && !skip_multisubset)
                {
                                 color_cell_compressor_params params5 = *pParams;

                                for ( uint32_t rotation = 0; rotation < 4; rotation++)
                                {
                                                if ((pComp_params->m_mode5_rotation_mask & (1 << rotation)) == 0)
                                                                continue;

                                                memcpy(params5.m_weights, pParams->m_weights, sizeof(params5.m_weights));
                                                if (rotation)
                                                                swapu(&params5.m_weights[rotation - 1], &params5.m_weights[3]);

                                                color_quad_i rot_pixels[16];
                                                const  color_quad_i * pTrial_pixels = pPixels;
                                                uint32_t trial_lo_a = 255, trial_hi_a = 255;
                                                if (rotation)
                                                {
                                                                trial_lo_a = 255;
                                                                trial_hi_a = 0;

                                                                for ( uint32_t i = 0; i < 16; i++)
                                                                {
                                                                                color_quad_i c = pPixels[i];
                                                                                swapi(&c.m_c[3], &c.m_c[rotation - 1]);
                                                                                rot_pixels[i] = c;

                                                                                trial_lo_a = minimumu(trial_lo_a, c.m_c[3]);
                                                                                trial_hi_a = maximumu(trial_hi_a, c.m_c[3]);
                                                                }

                                                                pTrial_pixels = rot_pixels;
                                                }

                                                bc7_optimization_results trial_opt_results5;

                                                uint64_t trial_mode5_err = 0;

                                                handle_alpha_block_mode5(pTrial_pixels, pComp_params, &params5, trial_lo_a, trial_hi_a, &trial_opt_results5, &trial_mode5_err);

                                                if (trial_mode5_err < best_err)
                                                {
                                                                best_err = trial_mode5_err;

                                                                opt_results = trial_opt_results5;
                                                                opt_results.m_rotation = rotation;
                                                }
                                } // rotation
                }

        // Mode 2 (3-subset, 5-bit color, 2-bit indices)
        // OPT-B: same rationale as Mode 0 — Mode 2 rarely wins (1.4% on
        // sheet.png) and costs 192 est() calls per try_3_subset block via
        // estimate_partition(2). Use a slightly lower threshold (128) since
        // Mode 2 wins more often than Mode 0 (5-bit endpoints vs 4-bit).
        const bool try_mode2 = try_3_subset && (best_err > 128);
        if (pComp_params->m_opaque_settings.m_use_mode[2] && !skip_multisubset && try_mode2)
        {
                // Bump the local array size so the adaptive logic has room to breathe if needed
                const uint32_t MAX_MODE2_ADAPTIVE = 16; 
                solution solutions3[MAX_MODE2_ADAPTIVE];
                uint32_t num_solutions3 = 0;
                
                if (pComp_params->m_opaque_settings.m_max_mode2_partitions_to_try == 1)
                {
                        solutions3[0].m_index = estimate_partition(2, pPixels, pComp_params);
                        num_solutions3 = 1;
                }
                else
                {
                        // ==========================================
                        // ADAPTIVE MODE 2: Be very aggressive here!
                        // Mode 2 only has 2-bit indices (blocky), so it 
                        // rarely wins anyway. Don't waste time on it.
                        // ==========================================
                        uint32_t dynamic_parts_2 = 4; // Default to extremely fast

                        if (pStats->luma_var > 15000) {
                                // Highly chaotic block, give it a fair shot
                                dynamic_parts_2 = 16; 
                        }
                        else if (pStats->luma_var > 5000) {
                                // Medium complexity
                                dynamic_parts_2 = 8; 
                        }
                        
                        // Ensure we never exceed the user's hardcoded max (usually 8)
                        dynamic_parts_2 = minimumu(dynamic_parts_2, pComp_params->m_opaque_settings.m_max_mode2_partitions_to_try);

                        part_score top_parts2[MAX_MODE2_ADAPTIVE];
                        find_top_mode2_partitions(pPixels, top_parts2, dynamic_parts_2);
                        
                        num_solutions3 = 0;
                        for (uint32_t i = 0; i < dynamic_parts_2; i++)
                        {
                                if (top_parts2[i].score == UINT64_MAX) continue;
                                
                                solutions3[num_solutions3].m_index = top_parts2[i].index;
                                num_solutions3++;
                        }
                        // ==========================================
                }

                                pParams->m_pSelector_weights = g_bc7_weights2;
                                pParams->m_pSelector_weightsx = (const vec4F *)&g_bc7_weights2x[0];
                                pParams->m_num_selector_weights = 4;

                                pParams->m_comp_bits = 5;
                                pParams->m_has_pbits = false;
                                pParams->m_endpoints_share_pbit = false;

                                pParams->m_perceptual = pComp_params->m_perceptual;

                                for ( uint32_t solution_index = 0; solution_index < num_solutions3; solution_index++)
                                {
                                                const int32_t best_partition2 = solutions3[solution_index].m_index;
                                                                                                
                                                uint32_t subset_total_colors2[3];
                                                subset_total_colors2[0] = 0;
                                                subset_total_colors2[1] = 0;
                                                subset_total_colors2[2] = 0;

                                                int subset_pixel_index2[3][16];
                                                                                                                
                                                const int *pPartition = &g_bc7_partition3[best_partition2 * 16];

                                                color_quad_i subset_colors[3][16];

                                                for ( uint32_t idx = 0; idx < 16; idx++)
                                                {
// #pragma ignore warning(perf)  [ISPC pragma, no-op in C++]
                                                                const uint32_t p = pPartition[idx];

// #pragma ignore warning(perf)  [ISPC pragma, no-op in C++]
                                                                subset_colors[p][subset_total_colors2[p]] = pPixels[idx];

// #pragma ignore warning(perf)  [ISPC pragma, no-op in C++]
                                                                subset_pixel_index2[p][subset_total_colors2[p]] = idx;

// #pragma ignore warning(perf)  [ISPC pragma, no-op in C++]
                                                                subset_total_colors2[p]++;
                                                }
                                                
                                                int subset_selectors2[3][16];
                                                color_cell_compressor_results subset_results2[3];
                                                                                                
                                                uint64_t mode2_err = 0;
                                                for ( uint32_t subset = 0; subset < 3; subset++)
                                                {
                                                                 color_cell_compressor_results * pResults = &subset_results2[subset];

                                                                pResults->m_pSelectors = &subset_selectors2[subset][0];
                                                                pResults->m_pSelectors_temp = selectors_temp;

                                                                uint64_t err = color_cell_compression(2, pParams, pResults, pComp_params, subset_total_colors2[subset], &subset_colors[subset][0], true);
                                                                assert(err == pResults->m_best_overall_err);

                                                                mode2_err += err;
                                                                if (mode2_err > best_err)
                                                                                break;
                                                } // subset

                                                if (mode2_err < best_err)
                                                {
                                                                best_err = mode2_err;

                                                                opt_results.m_mode = 2;
                                                                opt_results.m_index_selector = 0;
                                                                opt_results.m_rotation = 0;
                                                                opt_results.m_partition = best_partition2;

                                                                for ( uint32_t subset = 0; subset < 3; subset++)
                                                                {
                                                                                for ( uint32_t i = 0; i < subset_total_colors2[subset]; i++)
                                                                                {
                                                                                                const uint32_t pixel_index = subset_pixel_index2[subset][i];

// #pragma ignore warning(perf)  [ISPC pragma, no-op in C++]
                                                                                                opt_results.m_selectors[pixel_index] = subset_selectors2[subset][i];
                                                                                }

                                                                                opt_results.m_low[subset] = subset_results2[subset].m_low_endpoint;
                                                                                opt_results.m_high[subset] = subset_results2[subset].m_high_endpoint;
                                                                }
                                                }
                                }
                }

                // Mode 4 (1-subset, 5-bit RGB + 6-bit alpha, separate selectors, rotation)
                // For opaque blocks this uses rotation to treat an RGB channel as "alpha".
                if ((!pComp_params->m_perceptual) && (pComp_params->m_opaque_settings.m_use_mode[4]) && !skip_multisubset)
                {
                                 color_cell_compressor_params params4 = *pParams;

                                for ( uint32_t rotation = 0; rotation < 4; rotation++)
                                {
                                                if ((pComp_params->m_mode4_rotation_mask & (1 << rotation)) == 0)
                                                                continue;

                                                memcpy(params4.m_weights, pParams->m_weights, sizeof(params4.m_weights));
                                                if (rotation)
                                                                swapu(&params4.m_weights[rotation - 1], &params4.m_weights[3]);
                                                                                                                
                                                color_quad_i rot_pixels[16];
                                                const  color_quad_i * pTrial_pixels = pPixels;
                                                uint32_t trial_lo_a = 255, trial_hi_a = 255;
                                                if (rotation)
                                                {
                                                                trial_lo_a = 255;
                                                                trial_hi_a = 0;

                                                                for ( uint32_t i = 0; i < 16; i++)
                                                                {
                                                                                color_quad_i c = pPixels[i];
                                                                                swapi(&c.m_c[3], &c.m_c[rotation - 1]);
                                                                                rot_pixels[i] = c;

                                                                                trial_lo_a = minimumu(trial_lo_a, c.m_c[3]);
                                                                                trial_hi_a = maximumu(trial_hi_a, c.m_c[3]);
                                                                }

                                                                pTrial_pixels = rot_pixels;
                                                }

                                                bc7_optimization_results trial_opt_results4;

                                                uint64_t trial_mode4_err = best_err;

                                                handle_alpha_block_mode4(pTrial_pixels, pComp_params, &params4, trial_lo_a, trial_hi_a, &trial_opt_results4, &trial_mode4_err);

                                                if (trial_mode4_err < best_err)
                                                {
                                                                best_err = trial_mode4_err;
                                                        best_non_3subset_err = trial_mode4_err;

                                                                opt_results.m_mode = 4;
                                                                opt_results.m_index_selector = trial_opt_results4.m_index_selector;
                                                                opt_results.m_rotation = rotation;
                                                                opt_results.m_partition = 0;

                                                                opt_results.m_low[0] = trial_opt_results4.m_low[0];
                                                                opt_results.m_high[0] = trial_opt_results4.m_high[0];

                                                                for ( uint32_t i = 0; i < 16; i++)
                                                                                opt_results.m_selectors[i] = trial_opt_results4.m_selectors[i];
                                                                
                                                                for ( uint32_t i = 0; i < 16; i++)
                                                                                opt_results.m_alpha_selectors[i] = trial_opt_results4.m_alpha_selectors[i];
                                                }
                                } // rotation
                }

                if (best_err == UINT64_MAX)
                {
                        // No mode succeeded - force Mode 6 as fallback
                        pParams->m_pSelector_weights = g_bc7_weights4;
                        pParams->m_pSelector_weightsx = (const vec4F *)&g_bc7_weights4x[0];
                        pParams->m_num_selector_weights = 16;
                        pParams->m_comp_bits = 7;
                        pParams->m_has_pbits = true;
                        pParams->m_endpoints_share_pbit = false;
                        pParams->m_perceptual = pComp_params->m_perceptual;
                        
                        color_cell_compressor_results results6;
                        results6.m_pSelectors = opt_results.m_selectors;
                        results6.m_pSelectors_temp = selectors_temp;
                        best_err = color_cell_compression(6, pParams, &results6, pComp_params, 16, pPixels, true);
                        
                        opt_results.m_mode = 6;
                        opt_results.m_index_selector = 0;
                        opt_results.m_rotation = 0;
                        opt_results.m_partition = 0;
                        opt_results.m_low[0] = results6.m_low_endpoint;
                        opt_results.m_high[0] = results6.m_high_endpoint;
                        opt_results.m_pbits[0][0] = results6.m_pbits[0];
                        opt_results.m_pbits[0][1] = results6.m_pbits[1];
                }

        // FIX FM (Force Mode 6 for bias-prone Mode 1 opaque blocks):
        // Mode 1's 2-subset partition with 3-bit selectors can cause systematic
        // green DC bias from selector mismatches between subsets, visible as
        // "patched brightening" / "blocky darkening" on the green channel.
        // If Mode 1 won AND the green DC bias exceeds 1.5, force-encode as
        // Mode 6 (1-subset, 7-bit RGB + pbit = 8-bit, 16 selectors) to
        // eliminate the bias. Accept the forced Mode 6 only if:
        //   (1) its green DC bias is actually lower than Mode 1's, AND
        //   (2) its green SSE is not more than 1.5x Mode 1's (prevents the
        //       forced mode from creating worse max-error artifacts).
        if (opt_results.m_mode == 1 && pComp_params->m_opaque_settings.m_use_mode[6])
        {
                // Decode winning Mode 1 green DC (6-bit color + shared pbit = 7-bit)
                const int *pPart1 = &g_bc7_partition2[opt_results.m_partition * 16];
                int m1_dec_g = 0, m1_in_g = 0;
                uint64_t m1_sse_g = 0;
                for (uint32_t i = 0; i < 16; i++) {
                        uint32_t p = pPart1[i];
                        uint32_t sel = opt_results.m_selectors[i];
                        uint32_t w = g_bc7_weights3[sel];  // Mode 1 uses 3-bit selectors
                        uint32_t pbit = opt_results.m_pbits[p][0];  // shared pbit
                        uint32_t low_7bit  = (opt_results.m_low[p].m_c[1]  << 1) | pbit;
                        uint32_t high_7bit = (opt_results.m_high[p].m_c[1] << 1) | pbit;
                        // Scale 7-bit to 8-bit: (c << 1) | (c >> 6)
                        uint32_t aLowG  = (low_7bit  << 1) | (low_7bit  >> 6);
                        uint32_t aHighG = (high_7bit << 1) | (high_7bit >> 6);
                        int dg = (int)(((64 - w) * aLowG + w * aHighG + 32) >> 6);
                        m1_dec_g += dg;
                        m1_in_g  += pPixels[i].m_c[1];
                        int e = dg - pPixels[i].m_c[1];
                        m1_sse_g += (uint64_t)(e * e);
                }
                float m1_bias = (float)(m1_dec_g - m1_in_g) / 16.0f;

                if (m1_bias > 1.5f || m1_bias < -1.5f) {
                        // Force-encode as Mode 6 (1-subset, 7-bit + pbit = 8-bit)
                        color_cell_compressor_params params6 = *pParams;
                        params6.m_pSelector_weights = g_bc7_weights4;
                        params6.m_pSelector_weightsx = (const vec4F *)&g_bc7_weights4x[0];
                        params6.m_num_selector_weights = 16;
                        params6.m_comp_bits = 7;
                        params6.m_has_pbits = true;
                        params6.m_endpoints_share_pbit = false;
                        params6.m_has_alpha = false;
                        params6.m_perceptual = pComp_params->m_perceptual;

                        color_cell_compressor_results results6;
                        int selectors6[16];
                        results6.m_pSelectors = selectors6;
                        results6.m_pSelectors_temp = selectors_temp;
                        color_cell_compression(6, &params6, &results6, pComp_params, 16, pPixels, true);

                        // Compute Mode 6 green DC bias AND SSE (7-bit + pbit = 8-bit)
                        int m6_dec_g = 0;
                        uint64_t m6_sse_g = 0;
                        for (uint32_t i = 0; i < 16; i++) {
                                uint32_t sel = selectors6[i];
                                uint32_t w = g_bc7_weights4[sel];  // Mode 6 uses 4-bit selectors
                                uint32_t p0 = results6.m_pbits[0];
                                uint32_t p1 = results6.m_pbits[1];
                                uint32_t aLowG  = (results6.m_low_endpoint.m_c[1]  << 1) | p0;
                                uint32_t aHighG = (results6.m_high_endpoint.m_c[1] << 1) | p1;
                                int dg = (int)(((64 - w) * aLowG + w * aHighG + 32) >> 6);
                                m6_dec_g += dg;
                                int e = dg - pPixels[i].m_c[1];
                                m6_sse_g += (uint64_t)(e * e);
                        }
                        float m6_bias = (float)(m6_dec_g - m1_in_g) / 16.0f;

                        // Accept Mode 6 only if: (1) reduces |bias|, AND (2) SSE <= 1.5x Mode 1
                        if (fabsf(m6_bias) < fabsf(m1_bias) && m6_sse_g <= (m1_sse_g * 3 + 1) / 2) {
                                opt_results.m_mode = 6;
                                opt_results.m_index_selector = 0;
                                opt_results.m_rotation = 0;
                                opt_results.m_partition = 0;
                                opt_results.m_low[0] = results6.m_low_endpoint;
                                opt_results.m_high[0] = results6.m_high_endpoint;
                                opt_results.m_pbits[0][0] = results6.m_pbits[0];
                                opt_results.m_pbits[0][1] = results6.m_pbits[1];
                                for (uint32_t i = 0; i < 16; i++) opt_results.m_selectors[i] = selectors6[i];
                        }
                }
        }

                encode_bc7_block(pBlock, &opt_results);
}

static void handle_opaque_block_mode6(void * pBlock, const  color_quad_i * pPixels, const  bc7e_compress_block_params * pComp_params,  color_cell_compressor_params * pParams)
{
                int selectors_temp[16];
                                
                bc7_optimization_results opt_results;
                                
                uint64_t best_err = UINT64_MAX;

                // Mode 6
                pParams->m_pSelector_weights = g_bc7_weights4;
                pParams->m_pSelector_weightsx = (const vec4F * )&g_bc7_weights4x[0];
                pParams->m_num_selector_weights = 16;

                pParams->m_comp_bits = 7;
                pParams->m_has_pbits = true;
                pParams->m_endpoints_share_pbit = false;

                pParams->m_perceptual = pComp_params->m_perceptual;
                                                                
                color_cell_compressor_results results6;                                         
                results6.m_pSelectors = opt_results.m_selectors;
                results6.m_pSelectors_temp = selectors_temp;

                best_err = color_cell_compression(6, pParams, &results6, pComp_params, 16, pPixels, true);
                                                                                                
                opt_results.m_mode = 6;
                opt_results.m_index_selector = 0;
                opt_results.m_rotation = 0;
                opt_results.m_partition = 0;

                opt_results.m_low[0] = results6.m_low_endpoint;
                opt_results.m_high[0] = results6.m_high_endpoint;

                opt_results.m_pbits[0][0] = results6.m_pbits[0];
                opt_results.m_pbits[0][1] = results6.m_pbits[1];
                                
                encode_bc7_block_mode6(pBlock, &opt_results);
}

void bc7e_compress_blocks( uint32_t num_blocks,  uint64_t *  pBlocks, const  uint32_t *  pPixelsRGBA, const  bc7e_compress_block_params *  pComp_params)
{
                if (!g_codec_initialized)
                {
                                // Caller has forgotten to initialize the codec, or another thread is still working on that. We can't continue.
                                // What do we do here?
                                assert(0);

                                memset(pBlocks, 0, num_blocks * 16);
                                return;
                }

                 color_cell_compressor_params params;
                color_cell_compressor_params_clear(&params);

                memcpy(params.m_weights, pComp_params->m_weights, sizeof(params.m_weights));

                assert(pComp_params->m_mode4_rotation_mask != 0);
                assert(pComp_params->m_mode4_index_mask != 0);
                assert(pComp_params->m_mode5_rotation_mask != 0);
                assert(pComp_params->m_uber1_mask != 0);
                
                for (int block_index = 0; block_index < num_blocks; block_index++)
                {
                                const  color_quad_u8 * pSrcPixels = &((const  color_quad_u8 *)(pPixelsRGBA))[block_index * 16];

                                 color_quad_i temp_pixels[16];

                                float lo_a = 255, hi_a = 0;
                                
                                for ( uint32_t i = 0; i < 16; i++)
                                {
// #pragma ignore warning(perf)  [ISPC pragma, no-op in C++]
                                                color_quad_u8 c = pSrcPixels[i];

                                                int r = c.m_c[0];
                                                int g = c.m_c[1];
                                                int b = c.m_c[2];
                                                int a = c.m_c[3];
                                                                                                                                                                                                
                                                temp_pixels[i].m_c[0] = r;
                                                temp_pixels[i].m_c[1] = g;
                                                temp_pixels[i].m_c[2] = b;
                                                temp_pixels[i].m_c[3] = a;

                                                float fa = a;

                                                lo_a = min(lo_a, fa);
                                                hi_a = max(hi_a, fa);
                                }

                                 uint64_t * pBlock = &pBlocks[block_index * 2];

                                const bool has_alpha = (lo_a < 255);

                                // --- Fast path: flat (all-same) blocks ---
                                // Skip all mode evaluation for uniform blocks - huge speedup on sprite sheets/UI textures.
                                {
                                                bool all_same = true;
                                                for (int fi = 1; fi < 16; fi++)
                                                {
                                                                if (temp_pixels[fi].m_c[0] != temp_pixels[0].m_c[0] ||
                                                                        temp_pixels[fi].m_c[1] != temp_pixels[0].m_c[1] ||
                                                                        temp_pixels[fi].m_c[2] != temp_pixels[0].m_c[2] ||
                                                                        temp_pixels[fi].m_c[3] != temp_pixels[0].m_c[3])
                                                                {
                                                                                all_same = false;
                                                                                break;
                                                                }
                                                }
                                                if (all_same) {
                                                                encode_bc7_flat_block_mode5(pBlock,
                                                                temp_pixels[0].m_c[0], temp_pixels[0].m_c[1],
                                                                temp_pixels[0].m_c[2], temp_pixels[0].m_c[3]);
                                                                continue;
                                                }
                                }

                                // ---- NEW: near-flat fast path ----
                                {
                                uint32_t rMin=255,rMax=0, gMin=255,gMax=0;
                                uint32_t bMin=255,bMax=0, aMin=255,aMax=0;
                                for (int i = 0; i < 16; i++) {
                                                uint32_t r=temp_pixels[i].m_c[0], g=temp_pixels[i].m_c[1];
                                                uint32_t b=temp_pixels[i].m_c[2], a=temp_pixels[i].m_c[3];
                                                if(r<rMin)rMin=r; if(r>rMax)rMax=r;
                                                if(g<gMin)gMin=g; if(g>gMax)gMax=g;
                                                if(b<bMin)bMin=b; if(b>bMax)bMax=b;
                                                if(a<aMin)aMin=a; if(a>aMax)aMax=a;
                                }
                                constexpr uint32_t NEAR_FLAT_THRESH = 3;
                                if ((rMax-rMin <= NEAR_FLAT_THRESH) &&
                                                (gMax-gMin <= NEAR_FLAT_THRESH) &&
                                                (bMax-bMin <= NEAR_FLAT_THRESH) &&
                                                (aMax-aMin <= NEAR_FLAT_THRESH))
                                {
                                                encode_bc7_nearflat_block_mode5(pBlock, temp_pixels,
                                                rMin,rMax, gMin,gMax, bMin,bMax, aMin,aMax);
                                                continue;   // skip entire mode search
                                }
                                }

                                // ---- existing code continues ----
                                bc7_block_stats stats;
                                compute_block_stats(&stats, temp_pixels);

                                // --- Bimodal alpha fix for spritesheet frame boundaries ---
                                // When a 4x4 block straddles a frame edge, some pixels are opaque (frame
                                // content) and some are transparent (gap). Transparent pixels often have
                                // non-zero garbage RGB from the image pipeline. This poisons the color
                                // endpoints, causing brightening and alpha artifacts on opaque pixels.
                                // Fix: clamp nearly-transparent pixels to pure (0,0,0,0). This is visually
                                // lossless (alpha < 32 is invisible) and prevents garbage RGB from
                                // contaminating the color endpoint optimization.
                                if (has_alpha && stats.min_alpha < 32 && stats.max_alpha >= 224)
                                {
                                                for (int i = 0; i < 16; i++)
                                                {
                                                                if (temp_pixels[i].m_c[3] < 32)
                                                                {
                                                                                temp_pixels[i].m_c[0] = 0;
                                                                                temp_pixels[i].m_c[1] = 0;
                                                                                temp_pixels[i].m_c[2] = 0;
                                                                                temp_pixels[i].m_c[3] = 0;
                                                                }
                                                }
                                                lo_a = 0.0f;
                                                // hi_a unchanged - opaque pixels were not modified
                                }

                                const bool has_alpha_final = (lo_a < 255.0f);
                                const bool bimodal_alpha = (stats.min_alpha < 32 && stats.max_alpha >= 224);

                                if (has_alpha_final)
                                                handle_alpha_block(pBlock, temp_pixels, pComp_params, &params, (int)lo_a, (int)hi_a, &stats, bimodal_alpha);
                                else
                                {
                                                if (pComp_params->m_mode6_only)
                                                                handle_opaque_block_mode6(pBlock, temp_pixels, pComp_params, &params);
                                                else
                                                                handle_opaque_block(pBlock, temp_pixels, pComp_params, &params, &stats);
                                }
                }
}

void bc7e_compress_block_params_init(bc7e_compress_block_params *  p,  bool perceptual)
{
                p->m_max_partitions_mode[0] = BC7E_MAX_PARTITIONS0;
                p->m_max_partitions_mode[1] = BC7E_MAX_PARTITIONS1;
                p->m_max_partitions_mode[2] = BC7E_MAX_PARTITIONS2;
                p->m_max_partitions_mode[3] = BC7E_MAX_PARTITIONS3;
                p->m_max_partitions_mode[4] = 0;
                p->m_max_partitions_mode[5] = 0;
                p->m_max_partitions_mode[6] = 0;
                p->m_max_partitions_mode[7] = BC7E_MAX_PARTITIONS7;

                p->m_perceptual = perceptual;
                if (perceptual)
                {
                                p->m_weights[0] = 128;
                                p->m_weights[1] = 64;
                                p->m_weights[2] = 16;
                                p->m_weights[3] = 256;
                }
                else
                {
                                p->m_weights[0] = 1;
                                p->m_weights[1] = 1;
                                p->m_weights[2] = 1;
                                p->m_weights[3] = 1;
                }

                p->m_pbit_search = false;
                p->m_mode6_only = false;
                p->m_refinement_passes = 1;
                p->m_mode4_rotation_mask = 0xF;
                p->m_mode4_index_mask = 3;
                p->m_mode5_rotation_mask = 0xF;
                p->m_uber1_mask = 7;

                p->m_opaque_settings.m_use_mode[0] = false;
                p->m_opaque_settings.m_use_mode[1] = true;
                p->m_opaque_settings.m_use_mode[2] = true;
                p->m_opaque_settings.m_use_mode[3] = true;
                p->m_opaque_settings.m_use_mode[4] = false;
                p->m_opaque_settings.m_use_mode[5] = false;
                p->m_opaque_settings.m_use_mode[6] = true;
                p->m_opaque_settings.m_use_mode[7] = true;

                p->m_opaque_settings.m_max_mode13_partitions_to_try = 2;
                p->m_opaque_settings.m_max_mode0_partitions_to_try = 1;
                p->m_opaque_settings.m_max_mode2_partitions_to_try = 1;

                p->m_alpha_settings.m_use_mode4 = true;
                p->m_alpha_settings.m_use_mode5 = true;
                p->m_alpha_settings.m_use_mode6 = true;
                p->m_alpha_settings.m_use_mode7 = true;
                p->m_alpha_settings.m_use_mode4_rotation = true;
                p->m_alpha_settings.m_use_mode5_rotation = true;
                p->m_alpha_settings.m_max_mode7_partitions_to_try = 2;
                p->m_alpha_settings.m_mode67_error_weight_mul[0] = 1;
                p->m_alpha_settings.m_mode67_error_weight_mul[1] = 1;
                p->m_alpha_settings.m_mode67_error_weight_mul[2] = 1;
                p->m_alpha_settings.m_mode67_error_weight_mul[3] = 1;
                p->m_uber_level = 0;

                p->m_alpha_settings.m_use_mode4_rotation = false;
}

void bc7e_compress_block_params_init_slowest(bc7e_compress_block_params *  p,  bool perceptual)
{
                bc7e_compress_block_params_init(p, perceptual);

                p->m_opaque_settings.m_max_mode13_partitions_to_try = 4;
                p->m_opaque_settings.m_max_mode0_partitions_to_try = 4;
                p->m_opaque_settings.m_max_mode2_partitions_to_try = 4;
                p->m_alpha_settings.m_max_mode7_partitions_to_try = 4;

                p->m_pbit_search = true;
                p->m_uber_level = 4;
}

void bc7e_compress_block_params_init_veryslow(bc7e_compress_block_params *  p,  bool perceptual)
{
                bc7e_compress_block_params_init(p, perceptual);

                p->m_opaque_settings.m_max_mode13_partitions_to_try = 2;
                p->m_opaque_settings.m_max_mode0_partitions_to_try = 2;
                p->m_opaque_settings.m_max_mode2_partitions_to_try = 2;
                p->m_alpha_settings.m_max_mode7_partitions_to_try = 2;

                p->m_pbit_search = true;
                p->m_uber_level = 2;
}

void bc7e_compress_block_params_init_slow(bc7e_compress_block_params *  p,  bool perceptual)
{
                bc7e_compress_block_params_init(p, perceptual);

                p->m_alpha_settings.m_max_mode7_partitions_to_try = 2;

                p->m_pbit_search = true;
                p->m_uber_level = 0;
}

void bc7e_compress_block_params_init_basic(bc7e_compress_block_params *  p,  bool perceptual)
{
                bc7e_compress_block_params_init(p, perceptual);

                if (perceptual)
                {
                                p->m_opaque_settings.m_use_mode[0] = false;
                                p->m_opaque_settings.m_use_mode[2] = false;
                                p->m_opaque_settings.m_use_mode[3] = false;
                                p->m_opaque_settings.m_use_mode[4] = false;
                                p->m_opaque_settings.m_use_mode[5] = false;
                                
                                p->m_pbit_search = false;
                                p->m_uber_level = 1;
                }
                else
                {
                                p->m_max_partitions_mode[1] = 32;
                                p->m_max_partitions_mode[2] = 32;
                                p->m_max_partitions_mode[3] = 32;
                                p->m_max_partitions_mode[7] = 32;

                                p->m_opaque_settings.m_use_mode[2] = false;
                                
                                p->m_pbit_search = false;
                                p->m_uber_level = 1;
                }
}

void bc7e_compress_block_params_init_fast(bc7e_compress_block_params *  p,  bool perceptual)
{
                bc7e_compress_block_params_init(p, perceptual);

                if (perceptual)
                {
                                p->m_opaque_settings.m_use_mode[0] = false;
                                p->m_opaque_settings.m_use_mode[2] = false;
                                p->m_opaque_settings.m_use_mode[3] = false;
                                p->m_opaque_settings.m_use_mode[4] = false;
                                p->m_opaque_settings.m_use_mode[5] = false;

                                p->m_alpha_settings.m_use_mode5 = false;
                
                                p->m_opaque_settings.m_max_mode13_partitions_to_try = 1;
                                
                                p->m_pbit_search = false;
                                p->m_uber_level = 0;
                }               
                else
                {
                                p->m_opaque_settings.m_use_mode[0] = false;
                                p->m_opaque_settings.m_use_mode[2] = false;
                                p->m_opaque_settings.m_use_mode[4] = false;
                                p->m_opaque_settings.m_use_mode[5] = false;

                                p->m_alpha_settings.m_use_mode5 = false;
                
                                p->m_opaque_settings.m_max_mode13_partitions_to_try = 2;
                                
                                p->m_pbit_search = false;
                                p->m_uber_level = 0;
                }
}

void bc7e_compress_block_params_init_veryfast(bc7e_compress_block_params *  p,  bool perceptual)
{
                bc7e_compress_block_params_init(p, perceptual);

                if (perceptual)
                {
                                p->m_opaque_settings.m_use_mode[0] = false;
                                p->m_opaque_settings.m_use_mode[2] = false;
                                p->m_opaque_settings.m_use_mode[3] = false;
                                p->m_opaque_settings.m_use_mode[4] = false;
                                p->m_opaque_settings.m_use_mode[5] = false;

                                p->m_alpha_settings.m_use_mode5 = false;
                                
                                p->m_pbit_search = false;
                                p->m_uber_level = 0;
                }
                else
                {
                                p->m_opaque_settings.m_use_mode[2] = false;
                                p->m_opaque_settings.m_use_mode[4] = false;
                                p->m_opaque_settings.m_use_mode[5] = false;

                                p->m_alpha_settings.m_use_mode5 = false;
                                
                                p->m_pbit_search = false;
                                p->m_uber_level = 0;
                }
}

void bc7e_compress_block_params_init_ultrafast(bc7e_compress_block_params *  p,  bool perceptual)
{
                bc7e_compress_block_params_init(p, perceptual);

                p->m_mode6_only = true;

                p->m_alpha_settings.m_use_mode4 = true;
                p->m_alpha_settings.m_use_mode5 = true;
                p->m_alpha_settings.m_use_mode7 = false;

                p->m_mode4_rotation_mask = 1+4;
                p->m_mode4_index_mask = 3;
                p->m_mode5_rotation_mask = 1;
                                                                
                p->m_pbit_search = false;
                p->m_uber_level = 0;
}

// Thread-safe: compresses blocks[first..first+count) using pre-initialized codec.
// Each call uses only local state (params on stack, no global writes).
// The caller must ensure non-overlapping block ranges across threads.
static void bc7e_compress_block_range(
                uint32_t first_block, uint32_t count,
                uint64_t * pBlocks, const uint32_t * pPixelsRGBA,
                const bc7e_compress_block_params * pComp_params)
{
                if (!g_codec_initialized)
                {
                                memset(pBlocks, 0, count * 16);
                                return;
                }

                color_cell_compressor_params params;
                color_cell_compressor_params_clear(&params);
                memcpy(params.m_weights, pComp_params->m_weights, sizeof(params.m_weights));

                for (uint32_t bi = 0; bi < count; bi++)
                {
                                uint32_t block_index = first_block + bi;
                                const color_quad_u8 * pSrcPixels = &((const color_quad_u8 *)(pPixelsRGBA))[block_index * 16];
                                color_quad_i temp_pixels[16];
                                float lo_a = 255, hi_a = 0;
                                for (uint32_t i = 0; i < 16; i++)
                                {
                                                color_quad_u8 c = pSrcPixels[i];
                                                temp_pixels[i].m_c[0] = c.m_c[0];
                                                temp_pixels[i].m_c[1] = c.m_c[1];
                                                temp_pixels[i].m_c[2] = c.m_c[2];
                                                temp_pixels[i].m_c[3] = c.m_c[3];
                                                float fa = c.m_c[3];
                                                lo_a = min(lo_a, fa);
                                                hi_a = max(hi_a, fa);
                                }

                                uint64_t * pBlock = &pBlocks[block_index * 2];
                                const bool has_alpha = (lo_a < 255);

                                // --- LAST DITCH: 100% Transparent Fast Path ---
                                if (hi_a == 0.0f) {
                                                encode_bc7_flat_block_mode5(pBlock, 0, 0, 0, 0);
                                                continue; // Skip ALL math, stats, and mode evaluation
                                }

                                // Flat block fast path
                                {
                                                bool all_same = true;
                                                for (int fi = 1; fi < 16; fi++)
                                                {
                                                                if (temp_pixels[fi].m_c[0] != temp_pixels[0].m_c[0] ||
                                                                        temp_pixels[fi].m_c[1] != temp_pixels[0].m_c[1] ||
                                                                        temp_pixels[fi].m_c[2] != temp_pixels[0].m_c[2] ||
                                                                        temp_pixels[fi].m_c[3] != temp_pixels[0].m_c[3])
                                                                {
                                                                                all_same = false;
                                                                                break;
                                                                }
                                                }
                                                if (all_same)
                                                {
                                                                encode_bc7_flat_block_mode5(pBlock,
                                                                                temp_pixels[0].m_c[0], temp_pixels[0].m_c[1],
                                                                                temp_pixels[0].m_c[2], temp_pixels[0].m_c[3]);
                                                                continue;
                                                }
                                }

                                bc7_block_stats stats;
                                compute_block_stats(&stats, temp_pixels);

                                const bool has_alpha_final = (lo_a < 255.0f);
                                const bool bimodal_alpha = (stats.min_alpha < 64 && stats.max_alpha >= 128);

                                if (has_alpha_final)
                                                handle_alpha_block(pBlock, temp_pixels, pComp_params, &params, (int)lo_a, (int)hi_a, &stats, bimodal_alpha);
                                else
                                {
                                                if (pComp_params->m_mode6_only)
                                                                handle_opaque_block_mode6(pBlock, temp_pixels, pComp_params, &params);
                                                else
                                                                handle_opaque_block(pBlock, temp_pixels, pComp_params, &params, &stats);
                                }
                }
}

#endif // BC7_ENCODER_BASE_H


