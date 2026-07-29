// m7to5_nodecode.h - M7TO5 post-encode bias fix with NO DECODING.
//
// Bias is computed EXACTLY by reading the encoded bits (endpoints + pbits +
// partition + indexes) and applying the BC7 interpolation formula directly.
// The bc7decomp library is NOT invoked anywhere in this pass.
//
// The partition/anchor/weight tables are BC7 spec constants, inlined here
// so this header has zero external dependencies (only needs <cstdint>,
// <cstring>, <cmath>, <thread>, <atomic>, <vector>, <algorithm>, <cstdio>,
// <cstdlib> and the encoder's bc7e_compress_block_range / bc7e_compress_block_params).
//
// ============================================================================
// REFINEMENT CONTROLS (env-var tunable, sensible defaults)
// ============================================================================
// The post-encode pass is both SELECTIVE and STRICT:
//
//   1. PERFECT SHAPE early-exit (M7_PERFECT_EPS, default 0.20)
//      Blocks whose |green_bias| is already below this are NEVER touched.
//      Prevents the "already-good block gets slightly worse" regression
//      (e.g. the bottom of a glowing up-arrow becoming blocky).
//
//   2. TRIGGER threshold (M7_BIAS, default 0.50)
//      Blocks with |green_bias| above this are CONSIDERED for re-encoding.
//      Lower = more aggressive (more candidates). The old default was 1.0;
//      0.5 catches roughly 2x more blocks while still skipping near-perfect.
//
//   3. MIN_IMPROVE gate (M7_MIN_IMPROVE, default 0.25)
//      A candidate (M5 or M6) is only accepted if its |bias| is at least
//      (1 - MIN_IMPROVE) * |bias_orig| — i.e. it must reduce |bias| by >= 25%.
//      This rejects "unnecessary" swaps where the new mode barely helps.
//
//   4. SSE guard (M7_SSE_TOL_PCT, default 30; M7_PERFECT_SSE, default 50)
//      A candidate is rejected if its RGB SSE exceeds
//        sse_orig * (1 + SSE_TOL_PCT/100)
//      AND, if sse_orig < PERFECT_SSE, the candidate must have SSE <= sse_orig
//      (no increase allowed for already-accurate blocks). This catches the
//      "brightening artifact" where M5/M6 re-encoding shifts decoded values
//      lighter than the input.
//
//   5. Tiebreak by SSE (lower wins) when both M5 and M6 are valid.
//
// All five knobs are overridable via env vars without recompiling, so the
// user can A/B tune rapidly. The defaults are calibrated for the user's
// current test set (dad.png perf, sheet.png + 113x111 region quality, and
// the new arrow test subject).
// ============================================================================

#pragma once

#include <cstdint>
#include <cstring>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <atomic>
#include <vector>
#include <algorithm>

namespace m7to5_nodecode {

// g_bc7_partition2[64 * 16]: for each of 64 partition patterns, which subset
// (0 or 1) each of the 16 pixels belongs to.
static const uint8_t PARTITION2[64 * 16] = {
    0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,1,1,1,0,1,1,1,0,1,1,1,0,1,1,1,0,0,0,1,0,0,1,1,0,0,1,1,0,1,1,1,
    0,0,0,0,0,0,0,1,0,0,0,1,0,0,1,1,0,0,1,1,0,1,1,1,0,1,1,1,1,1,1,1,0,0,0,1,0,0,1,1,0,1,1,1,1,1,1,1,0,0,0,0,0,0,0,1,0,0,1,1,0,1,1,1,
    0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,1,0,0,1,1,0,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,1,0,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,1,0,1,1,1,
    0,0,0,1,0,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,
    0,0,0,0,1,0,0,0,1,1,1,0,1,1,1,1,0,1,1,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,1,1,0,0,1,1,1,0,0,1,1,0,0,0,1,0,0,0,0,
    0,0,1,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,1,0,0,1,1,1,0,0,0,0,0,0,0,0,0,1,0,0,0,1,1,0,0,0,1,1,1,0,0,1,1,0,0,1,1,0,0,0,1,
    0,0,1,1,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,1,1,0,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,0,1,1,0,1,1,0,0,1,1,0,1,1,0,0,
    0,0,0,1,0,1,1,1,1,1,1,0,1,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,1,1,1,0,0,0,1,1,0,0,0,1,1,1,0,0,0,1,1,1,0,0,1,1,0,0,1,1,1,0,0,
    0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1,0,1,0,1,1,0,1,0,0,1,0,1,1,0,1,0,0,0,1,1,0,0,1,1,1,1,0,0,1,1,0,0,
    0,0,1,1,1,1,0,0,0,0,1,1,1,1,0,0,0,1,0,1,0,1,0,1,1,0,1,0,1,0,1,0,0,1,1,0,1,0,0,1,0,1,1,0,1,0,0,1,0,1,0,1,1,0,1,0,1,0,1,0,0,1,0,1,
    0,1,1,1,0,0,1,1,1,1,0,0,1,1,1,0,0,0,0,1,0,0,1,1,1,1,0,0,1,0,0,0,0,0,1,1,0,0,1,0,0,1,0,0,1,1,0,0,0,0,1,1,1,0,1,1,1,1,0,1,1,1,0,0,
    0,1,1,0,1,0,0,1,1,0,0,1,0,1,1,0,0,0,1,1,1,1,0,0,1,1,0,0,0,0,1,1,0,1,1,0,0,1,1,0,1,0,0,1,1,0,0,1,0,0,0,0,0,1,1,0,0,1,1,0,0,0,0,0,
    0,1,0,0,1,1,1,0,0,1,0,0,0,0,0,0,0,0,1,0,0,1,1,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,1,1,0,0,1,0,0,0,0,0,0,1,0,0,1,1,1,0,0,1,0,0,
    0,1,1,0,1,1,0,0,1,0,0,1,0,0,1,1,0,0,1,1,0,1,1,0,1,1,0,0,1,0,0,1,0,1,1,0,0,0,1,1,1,0,0,1,1,1,0,0,0,0,1,1,1,0,0,1,1,1,0,0,0,1,1,0,
    0,1,1,0,1,1,0,0,1,1,0,0,1,0,0,1,0,1,1,0,0,0,1,1,0,0,1,1,1,0,0,1,0,1,1,1,1,1,1,0,1,0,0,0,0,0,0,1,0,0,0,1,1,0,0,0,1,1,1,0,0,1,1,1,
    0,0,0,0,1,1,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,1,1,1,1,0,0,0,0,0,0,1,0,0,0,1,0,1,1,1,0,1,1,1,0,0,1,0,0,0,1,0,0,0,1,1,1,0,1,1,1,
};

// g_bc7_table_anchor_index_second_subset[64]: index of the anchor pixel for
// the second subset (its index is encoded with 1 fewer bit).
static const uint8_t ANCHOR_SECOND_SUBSET[64] = {
    15,15,15,15,15,15,15,15,    15,15,15,15,15,15,15,15,
    15, 2, 8, 2, 2, 8, 8,15,     2, 8, 2, 2, 8, 8, 2, 2,
    15,15, 6, 8, 2, 8,15,15,     2, 8, 2, 2, 2,15,15, 6,
     6, 2, 6, 8,15,15, 2, 2,    15,15,15,15,15, 2, 2,15
};

// BC7 interpolation weight tables (unnormalized; divide by 64).
static const uint32_t WEIGHTS2[4]   = { 0, 21, 43, 64 };  // 2-bit indexes
static const uint32_t WEIGHTS3[8]   = { 0, 9, 18, 27, 37, 46, 55, 64 };  // 3-bit indexes
static const uint32_t WEIGHTS4[16]  = { 0, 4, 9, 13, 17, 21, 26, 30, 34, 38, 43, 47, 51, 55, 60, 64 };  // 4-bit indexes

// Generic LSB-first bit reader for a 16-byte BC7 block.
static inline uint32_t read_bits(const uint8_t* p, uint32_t bit_offset, uint32_t codesize) {
    uint32_t bits = 0, total = 0;
    while (total < codesize) {
        uint32_t byte_bit = bit_offset & 7;
        uint32_t to_read = (codesize - total < 8u - byte_bit) ? (codesize - total) : (8u - byte_bit);
        uint32_t b = p[bit_offset >> 3] >> byte_bit;
        b &= ((1u << to_read) - 1);
        bits |= (b << total);
        total += to_read;
        bit_offset += to_read;
    }
    return bits;
}

static inline int detect_mode(const void* pBlock) {
    uint8_t b = ((const uint8_t*)pBlock)[0];
    for (int m = 0; m <= 7; m++) {
        if (b & (1u << m)) return m;
    }
    return -1;
}

static inline int sum_input_green(const uint8_t* in_pix_16) {
    int s = 0;
    for (int i = 0; i < 16; i++) s += in_pix_16[i * 4 + 1];
    return s;
}

// Dequantize N-bit endpoint value to 8-bit (BC7 style).
static inline uint32_t dequant_no_pbit(uint32_t v, uint32_t val_bits) {
    v <<= (8 - val_bits);
    v |= (v >> val_bits);
    return v;
}
static inline uint32_t dequant_with_pbit(uint32_t v, uint32_t p, uint32_t val_bits) {
    uint32_t total = val_bits + 1;
    v = (v << 1) | p;
    v <<= (8 - total);
    v |= (v >> total);
    return v;
}

// Interpolate (l, h) at weight w (0..64). Same formula bc7decomp uses.
static inline uint32_t interp(uint32_t l, uint32_t h, uint32_t w) {
    return (l * (64 - w) + h * w + 32) >> 6;
}

// ---------------------------------------------------------------------------
// Decoded RGBA for 16 pixels. SSE guard includes alpha to catch cases where
// M5/M6 (1-subset) can't match M7's (2-subset) alpha precision.
// ---------------------------------------------------------------------------
struct RGB16 {
    uint8_t r[16];
    uint8_t g[16];
    uint8_t b[16];
    uint8_t a[16];  // alpha (added to catch alpha regressions in SSE guard)
};

// ---------------------------------------------------------------------------
// Mode 4: 1-subset, 5-bit RGB endpoints (no pbit), 6-bit A endpoints,
//         2-bit/3-bit indexes, comp_rot, index_mode.
// ---------------------------------------------------------------------------
static inline void mode4_decode_rgb(const uint8_t* pBlock, RGB16* out) {
    uint32_t comp_rot   = read_bits(pBlock, 5, 2);
    uint32_t index_mode = read_bits(pBlock, 7, 1);

    // Endpoints (5-bit RGB, 6-bit A)
    uint32_t r0 = dequant_no_pbit(read_bits(pBlock,  8, 5), 5);
    uint32_t r1 = dequant_no_pbit(read_bits(pBlock, 13, 5), 5);
    uint32_t g0 = dequant_no_pbit(read_bits(pBlock, 18, 5), 5);
    uint32_t g1 = dequant_no_pbit(read_bits(pBlock, 23, 5), 5);
    uint32_t b0 = dequant_no_pbit(read_bits(pBlock, 28, 5), 5);
    uint32_t b1 = dequant_no_pbit(read_bits(pBlock, 33, 5), 5);
    uint32_t a0 = dequant_no_pbit(read_bits(pBlock, 38, 6), 6);
    uint32_t a1 = dequant_no_pbit(read_bits(pBlock, 44, 6), 6);

    // Weight sets:
    //   First set  = 2-bit (1-bit anchor at i=0)  -> indices into WEIGHTS2
    //   Second set = 3-bit (2-bit anchor at i=0)  -> indices into WEIGHTS3
    // index_mode=0: color = first set, alpha = second set
    // index_mode=1: alpha = first set, color = second set
    uint32_t first_set[16];
    uint32_t second_set[16];
    uint32_t bit_offset = 50;
    for (int i = 0; i < 16; i++) {
        uint32_t bits = (i == 0) ? 1 : 2;
        first_set[i] = read_bits(pBlock, bit_offset, bits);
        bit_offset += bits;
    }
    for (int i = 0; i < 16; i++) {
        uint32_t bits = (i == 0) ? 2 : 3;
        second_set[i] = read_bits(pBlock, bit_offset, bits);
        bit_offset += bits;
    }

    // Resolve which set is color vs alpha, and the corresponding weight table.
    // First set indices are always 2-bit (into WEIGHTS2).
    // Second set indices are always 3-bit (into WEIGHTS3).
    const uint32_t* color_w;
    const uint32_t* alpha_w;
    const uint32_t* color_tbl;  // table to look up color_w indices
    const uint32_t* alpha_tbl;  // table to look up alpha_w indices
    if (index_mode == 0) {
        color_w = first_set;   color_tbl = WEIGHTS2;
        alpha_w = second_set;  alpha_tbl = WEIGHTS3;
    } else {
        alpha_w = first_set;   alpha_tbl = WEIGHTS2;
        color_w = second_set;  color_tbl = WEIGHTS3;
    }

    // comp_rot: 0=identity, 1=R<->A, 2=G<->A, 3=B<->A
    // When a channel is swapped with A, it uses the ALPHA endpoint + ALPHA weights.
    // The ALPHA output then uses that channel's endpoint + COLOR weights.
    for (int i = 0; i < 16; i++) {
        uint32_t cw = color_tbl[color_w[i]];
        uint32_t aw = alpha_tbl[alpha_w[i]];
        // R
        if (comp_rot == 1) {
            out->r[i] = (uint8_t)interp(a0, a1, aw);
        } else {
            out->r[i] = (uint8_t)interp(r0, r1, cw);
        }
        // G
        if (comp_rot == 2) {
            out->g[i] = (uint8_t)interp(a0, a1, aw);
        } else {
            out->g[i] = (uint8_t)interp(g0, g1, cw);
        }
        // B
        if (comp_rot == 3) {
            out->b[i] = (uint8_t)interp(a0, a1, aw);
        } else {
            out->b[i] = (uint8_t)interp(b0, b1, cw);
        }
        // Alpha: swapped channel's endpoint + color weights when comp_rot != 0
        if (comp_rot == 0) {
            out->a[i] = (uint8_t)interp(a0, a1, aw);
        } else if (comp_rot == 1) {
            out->a[i] = (uint8_t)interp(r0, r1, cw);
        } else if (comp_rot == 2) {
            out->a[i] = (uint8_t)interp(g0, g1, cw);
        } else { // comp_rot == 3
            out->a[i] = (uint8_t)interp(b0, b1, cw);
        }
    }
}

// EXACT bias (green only) — kept fast path, no full RGB decode.
static inline double mode4_green_bias(const uint8_t* pBlock, const uint8_t* in_pix_16) {
    uint32_t comp_rot   = read_bits(pBlock, 5, 2);
    uint32_t index_mode = read_bits(pBlock, 7, 1);

    bool green_from_alpha = (comp_rot == 2);

    uint32_t e0_8, e1_8;
    if (green_from_alpha) {
        uint32_t a0 = read_bits(pBlock, 38, 6);
        uint32_t a1 = read_bits(pBlock, 44, 6);
        e0_8 = dequant_no_pbit(a0, 6);
        e1_8 = dequant_no_pbit(a1, 6);
    } else {
        uint32_t g0 = read_bits(pBlock, 18, 5);
        uint32_t g1 = read_bits(pBlock, 23, 5);
        e0_8 = dequant_no_pbit(g0, 5);
        e1_8 = dequant_no_pbit(g1, 5);
    }

    uint32_t bit_offset = 50;
    uint32_t weights[16];

    bool need_color_w = !green_from_alpha;
    bool first_is_color = (index_mode == 0);
    bool need_first_set = (need_color_w == first_is_color);

    uint32_t w_bits, w_anchor;
    const uint32_t* wtable;
    if (need_color_w) {
        if (index_mode == 0) { w_bits = 2; wtable = WEIGHTS2; }
        else                 { w_bits = 3; wtable = WEIGHTS3; }
    } else {
        if (index_mode == 0) { w_bits = 3; wtable = WEIGHTS3; }
        else                 { w_bits = 2; wtable = WEIGHTS2; }
    }
    w_anchor = w_bits - 1;

    if (!need_first_set) {
        bit_offset += 1 + 15 * 2;
    }
    for (int i = 0; i < 16; i++) {
        uint32_t bits = (i == 0) ? w_anchor : w_bits;
        weights[i] = read_bits(pBlock, bit_offset, bits);
        bit_offset += bits;
    }

    int sum_dec = 0;
    for (int i = 0; i < 16; i++) {
        sum_dec += interp(e0_8, e1_8, wtable[weights[i]]);
    }
    double avg_dec = sum_dec / 16.0;
    double avg_in  = sum_input_green(in_pix_16) / 16.0;
    return avg_dec - avg_in;
}

// ---------------------------------------------------------------------------
// Mode 5: 1-subset, 7-bit RGB endpoints (no pbit), 8-bit A endpoints,
//         2-bit indexes, comp_rot.
// ---------------------------------------------------------------------------
static inline void mode5_decode_rgb(const uint8_t* pBlock, RGB16* out) {
    uint32_t comp_rot = read_bits(pBlock, 6, 2);

    uint32_t r0 = dequant_no_pbit(read_bits(pBlock,  8, 7), 7);
    uint32_t r1 = dequant_no_pbit(read_bits(pBlock, 15, 7), 7);
    uint32_t g0 = dequant_no_pbit(read_bits(pBlock, 22, 7), 7);
    uint32_t g1 = dequant_no_pbit(read_bits(pBlock, 29, 7), 7);
    uint32_t b0 = dequant_no_pbit(read_bits(pBlock, 36, 7), 7);
    uint32_t b1 = dequant_no_pbit(read_bits(pBlock, 43, 7), 7);
    uint32_t a0 = read_bits(pBlock, 50, 8);  // 8-bit, no dequant
    uint32_t a1 = read_bits(pBlock, 58, 8);

    // Mode 5: both color and alpha weights are 2-bit (1-bit anchor at i=0).
    uint32_t color_w[16];
    uint32_t alpha_w[16];
    uint32_t bit_offset = 66;
    for (int i = 0; i < 16; i++) {
        uint32_t bits = (i == 0) ? 1 : 2;
        color_w[i] = read_bits(pBlock, bit_offset, bits);
        bit_offset += bits;
    }
    for (int i = 0; i < 16; i++) {
        uint32_t bits = (i == 0) ? 1 : 2;
        alpha_w[i] = read_bits(pBlock, bit_offset, bits);
        bit_offset += bits;
    }

    for (int i = 0; i < 16; i++) {
        // R
        if (comp_rot == 1) {
            out->r[i] = (uint8_t)interp(a0, a1, WEIGHTS2[alpha_w[i]]);
        } else {
            out->r[i] = (uint8_t)interp(r0, r1, WEIGHTS2[color_w[i]]);
        }
        // G
        if (comp_rot == 2) {
            out->g[i] = (uint8_t)interp(a0, a1, WEIGHTS2[alpha_w[i]]);
        } else {
            out->g[i] = (uint8_t)interp(g0, g1, WEIGHTS2[color_w[i]]);
        }
        // B
        if (comp_rot == 3) {
            out->b[i] = (uint8_t)interp(a0, a1, WEIGHTS2[alpha_w[i]]);
        } else {
            out->b[i] = (uint8_t)interp(b0, b1, WEIGHTS2[color_w[i]]);
        }
        // Alpha: comp_rot swaps alpha with one color channel.
        // comp_rot==0: A = interp(a0,a1, alpha_w)
        // comp_rot==1: A = interp(r0,r1, color_w)  (R and A swapped)
        // comp_rot==2: A = interp(g0,g1, color_w)
        // comp_rot==3: A = interp(b0,b1, color_w)
        if (comp_rot == 0) {
            out->a[i] = (uint8_t)interp(a0, a1, WEIGHTS2[alpha_w[i]]);
        } else if (comp_rot == 1) {
            out->a[i] = (uint8_t)interp(r0, r1, WEIGHTS2[color_w[i]]);
        } else if (comp_rot == 2) {
            out->a[i] = (uint8_t)interp(g0, g1, WEIGHTS2[color_w[i]]);
        } else { // comp_rot == 3
            out->a[i] = (uint8_t)interp(b0, b1, WEIGHTS2[color_w[i]]);
        }
    }
}

static inline double mode5_green_bias(const uint8_t* pBlock, const uint8_t* in_pix_16) {
    uint32_t comp_rot = read_bits(pBlock, 6, 2);
    bool green_from_alpha = (comp_rot == 2);

    uint32_t e0_8, e1_8;
    if (green_from_alpha) {
        e0_8 = read_bits(pBlock, 50, 8);
        e1_8 = read_bits(pBlock, 58, 8);
    } else {
        uint32_t g0 = read_bits(pBlock, 22, 7);
        uint32_t g1 = read_bits(pBlock, 29, 7);
        e0_8 = dequant_no_pbit(g0, 7);
        e1_8 = dequant_no_pbit(g1, 7);
    }

    uint32_t bit_offset = 66;
    if (green_from_alpha) bit_offset += 31;  // skip color weights

    uint32_t weights[16];
    for (int i = 0; i < 16; i++) {
        uint32_t bits = (i == 0) ? 1 : 2;
        weights[i] = read_bits(pBlock, bit_offset, bits);
        bit_offset += bits;
    }

    int sum_dec = 0;
    for (int i = 0; i < 16; i++) {
        sum_dec += interp(e0_8, e1_8, WEIGHTS2[weights[i]]);
    }
    double avg_dec = sum_dec / 16.0;
    double avg_in  = sum_input_green(in_pix_16) / 16.0;
    return avg_dec - avg_in;
}

// ---------------------------------------------------------------------------
// Mode 6: 1-subset, 7-bit RGBA endpoints, 1 pbit per endpoint (shared),
//         4-bit indexes.
// ---------------------------------------------------------------------------
static inline void mode6_decode_rgb(const uint8_t* pBlock, RGB16* out) {
    uint32_t r0 = read_bits(pBlock,  7, 7);
    uint32_t r1 = read_bits(pBlock, 14, 7);
    uint32_t g0 = read_bits(pBlock, 21, 7);
    uint32_t g1 = read_bits(pBlock, 28, 7);
    uint32_t b0 = read_bits(pBlock, 35, 7);
    uint32_t b1 = read_bits(pBlock, 42, 7);
    uint32_t p0 = read_bits(pBlock, 63, 1);
    uint32_t p1 = read_bits(pBlock, 64, 1);

    uint32_t r0_8 = dequant_with_pbit(r0, p0, 7);
    uint32_t r1_8 = dequant_with_pbit(r1, p1, 7);
    uint32_t g0_8 = dequant_with_pbit(g0, p0, 7);
    uint32_t g1_8 = dequant_with_pbit(g1, p1, 7);
    uint32_t b0_8 = dequant_with_pbit(b0, p0, 7);
    uint32_t b1_8 = dequant_with_pbit(b1, p1, 7);
    uint32_t a0_8 = dequant_with_pbit(read_bits(pBlock, 49, 7), p0, 7);
    uint32_t a1_8 = dequant_with_pbit(read_bits(pBlock, 56, 7), p1, 7);

    uint32_t bit_offset = 65;
    uint32_t weights[16];
    for (int i = 0; i < 16; i++) {
        uint32_t bits = (i == 0) ? 3 : 4;
        weights[i] = read_bits(pBlock, bit_offset, bits);
        bit_offset += bits;
    }

    for (int i = 0; i < 16; i++) {
        uint32_t w = WEIGHTS4[weights[i]];
        out->r[i] = (uint8_t)interp(r0_8, r1_8, w);
        out->g[i] = (uint8_t)interp(g0_8, g1_8, w);
        out->b[i] = (uint8_t)interp(b0_8, b1_8, w);
        out->a[i] = (uint8_t)interp(a0_8, a1_8, w);  // Mode 6: same 4-bit weights
    }
}

static inline double mode6_green_bias(const uint8_t* pBlock, const uint8_t* in_pix_16) {
    uint32_t g0 = read_bits(pBlock, 21, 7);
    uint32_t g1 = read_bits(pBlock, 28, 7);
    uint32_t p0 = read_bits(pBlock, 63, 1);
    uint32_t p1 = read_bits(pBlock, 64, 1);
    uint32_t e0_8 = dequant_with_pbit(g0, p0, 7);
    uint32_t e1_8 = dequant_with_pbit(g1, p1, 7);

    uint32_t bit_offset = 65;
    uint32_t weights[16];
    for (int i = 0; i < 16; i++) {
        uint32_t bits = (i == 0) ? 3 : 4;
        weights[i] = read_bits(pBlock, bit_offset, bits);
        bit_offset += bits;
    }

    int sum_dec = 0;
    for (int i = 0; i < 16; i++) {
        sum_dec += interp(e0_8, e1_8, WEIGHTS4[weights[i]]);
    }
    double avg_dec = sum_dec / 16.0;
    double avg_in  = sum_input_green(in_pix_16) / 16.0;
    return avg_dec - avg_in;
}

// ---------------------------------------------------------------------------
// Mode 7: 2-subset, 5-bit RGBA endpoints, 1 pbit per endpoint, 2-bit indexes.
// ---------------------------------------------------------------------------
static inline void mode7_decode_rgb(const uint8_t* pBlock, RGB16* out) {
    uint32_t part = read_bits(pBlock, 8, 6);

    uint32_t r[4], g[4], b[4], a[4], p[4];
    r[0] = read_bits(pBlock, 14, 5);
    r[1] = read_bits(pBlock, 19, 5);
    r[2] = read_bits(pBlock, 24, 5);
    r[3] = read_bits(pBlock, 29, 5);
    g[0] = read_bits(pBlock, 34, 5);
    g[1] = read_bits(pBlock, 39, 5);
    g[2] = read_bits(pBlock, 44, 5);
    g[3] = read_bits(pBlock, 49, 5);
    b[0] = read_bits(pBlock, 54, 5);
    b[1] = read_bits(pBlock, 59, 5);
    b[2] = read_bits(pBlock, 64, 5);
    b[3] = read_bits(pBlock, 69, 5);
    a[0] = read_bits(pBlock, 74, 5);
    a[1] = read_bits(pBlock, 79, 5);
    a[2] = read_bits(pBlock, 84, 5);
    a[3] = read_bits(pBlock, 89, 5);
    p[0] = read_bits(pBlock, 94, 1);
    p[1] = read_bits(pBlock, 95, 1);
    p[2] = read_bits(pBlock, 96, 1);
    p[3] = read_bits(pBlock, 97, 1);

    uint32_t r8[4], g8[4], b8[4], a8[4];
    for (int e = 0; e < 4; e++) {
        r8[e] = dequant_with_pbit(r[e], p[e], 5);
        g8[e] = dequant_with_pbit(g[e], p[e], 5);
        b8[e] = dequant_with_pbit(b[e], p[e], 5);
        a8[e] = dequant_with_pbit(a[e], p[e], 5);
    }

    uint32_t anchor2 = ANCHOR_SECOND_SUBSET[part];
    uint32_t bit_offset = 98;
    uint32_t weights[16];
    for (int i = 0; i < 16; i++) {
        uint32_t bits = (i == 0 || i == (int)anchor2) ? 1 : 2;
        weights[i] = read_bits(pBlock, bit_offset, bits);
        bit_offset += bits;
    }

    for (int i = 0; i < 16; i++) {
        int subset = PARTITION2[part * 16 + i];
        uint32_t e_lo = subset * 2 + 0;
        uint32_t e_hi = subset * 2 + 1;
        uint32_t w = WEIGHTS2[weights[i]];
        out->r[i] = (uint8_t)interp(r8[e_lo], r8[e_hi], w);
        out->g[i] = (uint8_t)interp(g8[e_lo], g8[e_hi], w);
        out->b[i] = (uint8_t)interp(b8[e_lo], b8[e_hi], w);
        out->a[i] = (uint8_t)interp(a8[e_lo], a8[e_hi], w);  // Mode 7: same 2-bit weights
    }
}

static inline double mode7_green_bias(const uint8_t* pBlock, const uint8_t* in_pix_16) {
    uint32_t part = read_bits(pBlock, 8, 6);

    uint32_t g[4], p[4];
    g[0] = read_bits(pBlock, 34, 5);
    g[1] = read_bits(pBlock, 39, 5);
    g[2] = read_bits(pBlock, 44, 5);
    g[3] = read_bits(pBlock, 49, 5);
    p[0] = read_bits(pBlock, 94, 1);
    p[1] = read_bits(pBlock, 95, 1);
    p[2] = read_bits(pBlock, 96, 1);
    p[3] = read_bits(pBlock, 97, 1);

    uint32_t g8[4];
    for (int e = 0; e < 4; e++) g8[e] = dequant_with_pbit(g[e], p[e], 5);

    uint32_t anchor2 = ANCHOR_SECOND_SUBSET[part];
    uint32_t bit_offset = 98;
    uint32_t weights[16];
    for (int i = 0; i < 16; i++) {
        uint32_t bits = (i == 0 || i == (int)anchor2) ? 1 : 2;
        weights[i] = read_bits(pBlock, bit_offset, bits);
        bit_offset += bits;
    }

    int sum_dec = 0;
    for (int i = 0; i < 16; i++) {
        int subset = PARTITION2[part * 16 + i];
        uint32_t e_lo = g8[subset * 2 + 0];
        uint32_t e_hi = g8[subset * 2 + 1];
        sum_dec += interp(e_lo, e_hi, WEIGHTS2[weights[i]]);
    }
    double avg_dec = sum_dec / 16.0;
    double avg_in  = sum_input_green(in_pix_16) / 16.0;
    return avg_dec - avg_in;
}

// ---------------------------------------------------------------------------
// RGBA SSE between decoded block and input pixels (sum of squared errors
// across all 16 pixels x 4 channels). Includes alpha to catch cases where
// M5/M6 (1-subset) can't match M7's (2-subset) alpha precision.
// ---------------------------------------------------------------------------
static inline double mode4_rgb_sse(const uint8_t* pBlock, const uint8_t* in_pix_16) {
    RGB16 dec;
    mode4_decode_rgb(pBlock, &dec);
    long sse = 0;
    for (int i = 0; i < 16; i++) {
        int dr = (int)dec.r[i] - (int)in_pix_16[i * 4 + 0];
        int dg = (int)dec.g[i] - (int)in_pix_16[i * 4 + 1];
        int db = (int)dec.b[i] - (int)in_pix_16[i * 4 + 2];
        int da = (int)dec.a[i] - (int)in_pix_16[i * 4 + 3];
        sse += dr * dr + dg * dg + db * db + da * da;
    }
    return (double)sse;
}
static inline double mode5_rgb_sse(const uint8_t* pBlock, const uint8_t* in_pix_16) {
    RGB16 dec;
    mode5_decode_rgb(pBlock, &dec);
    long sse = 0;
    for (int i = 0; i < 16; i++) {
        int dr = (int)dec.r[i] - (int)in_pix_16[i * 4 + 0];
        int dg = (int)dec.g[i] - (int)in_pix_16[i * 4 + 1];
        int db = (int)dec.b[i] - (int)in_pix_16[i * 4 + 2];
        int da = (int)dec.a[i] - (int)in_pix_16[i * 4 + 3];
        sse += dr * dr + dg * dg + db * db + da * da;
    }
    return (double)sse;
}
static inline double mode6_rgb_sse(const uint8_t* pBlock, const uint8_t* in_pix_16) {
    RGB16 dec;
    mode6_decode_rgb(pBlock, &dec);
    long sse = 0;
    for (int i = 0; i < 16; i++) {
        int dr = (int)dec.r[i] - (int)in_pix_16[i * 4 + 0];
        int dg = (int)dec.g[i] - (int)in_pix_16[i * 4 + 1];
        int db = (int)dec.b[i] - (int)in_pix_16[i * 4 + 2];
        int da = (int)dec.a[i] - (int)in_pix_16[i * 4 + 3];
        sse += dr * dr + dg * dg + db * db + da * da;
    }
    return (double)sse;
}
static inline double mode7_rgb_sse(const uint8_t* pBlock, const uint8_t* in_pix_16) {
    RGB16 dec;
    mode7_decode_rgb(pBlock, &dec);
    long sse = 0;
    for (int i = 0; i < 16; i++) {
        int dr = (int)dec.r[i] - (int)in_pix_16[i * 4 + 0];
        int dg = (int)dec.g[i] - (int)in_pix_16[i * 4 + 1];
        int db = (int)dec.b[i] - (int)in_pix_16[i * 4 + 2];
        int da = (int)dec.a[i] - (int)in_pix_16[i * 4 + 3];
        sse += dr * dr + dg * dg + db * db + da * da;
    }
    return (double)sse;
}

// Build a single-mode-only params override.
// (Kept for backwards compatibility — run() no longer calls the encoder for
//  re-encoding. Manual builders below produce M5/M6 blocks directly.)
static inline void make_single_mode_params(
    const bc7e_compress_block_params* base,
    bc7e_compress_block_params* out,
    int target_mode)
{
    *out = *base;
    for (int m = 0; m < 7; m++) {
        out->m_opaque_settings.m_use_mode[m] = (m == target_mode) ? 1 : 0;
    }
    out->m_opaque_settings.m_unused1 = (target_mode == 7) ? 1 : 0;
    out->m_alpha_settings.m_use_mode4 = (target_mode == 4) ? 1 : 0;
    out->m_alpha_settings.m_use_mode5 = (target_mode == 5) ? 1 : 0;
    out->m_alpha_settings.m_use_mode6 = (target_mode == 6) ? 1 : 0;
    out->m_alpha_settings.m_use_mode7 = (target_mode == 7) ? 1 : 0;
}

// ---------------------------------------------------------------------------
// LSB-first bit writer for a 16-byte BC7 block.
// ---------------------------------------------------------------------------
static inline void put_bits(uint8_t* pBlock, uint32_t bit_offset,
                            uint32_t codesize, uint32_t value) {
    for (uint32_t i = 0; i < codesize; i++) {
        if (value & (1u << i)) {
            uint32_t b = bit_offset + i;
            pBlock[b >> 3] |= (1u << (b & 7));
        }
    }
}

static inline uint32_t quant7(int v8) {  // 8-bit (0..255) -> 7-bit (0..127)
    return (uint32_t)((v8 * 127 + 127) / 255);
}
static inline uint32_t dq7(uint32_t v7) {  // 7-bit -> 8-bit (no pbit)
    // Must match dequant_no_pbit(v, 7): shift first, then OR with shifted value.
    // (v7 << 1) | (v7 >> 7) is WRONG — it ORs with the original v7, not shifted.
    uint32_t t = v7 << 1;
    return t | (t >> 7);
}
static inline uint32_t dq7p(uint32_t v7, uint32_t p) {  // 7-bit + pbit -> 8-bit
    // For val_bits=7, total=8: dequant_with_pbit formula reduces to just
    // (v7 << 1) | p  (the <<= 0 and |= (v>>8) are no-ops).
    return ((v7 << 1) | (p & 1)) & 0xFF;
}

// ---------------------------------------------------------------------------
// Manual Mode 5 builder.
// SSE-optimized: endpoints = min/max of input (quantized to 7-bit RGB, 8-bit A).
// 2-bit weights picked per-pixel to minimize RGB+alpha error.
// Anchor at i=0 is restricted to {0, 1} (MSB forced to 0 per BC7 spec).
//
// For smooth blocks, min/max endpoints naturally give low DC bias because
// avg(dequant(min), dequant(max)) ≈ avg(input). For high-contrast blocks,
// the SSE guard in run() will reject this candidate (M7 is better there).
//
// This bypasses the encoder's max_color_range<128 guard that was rejecting
// 49/51 high-contrast candidates.
// ---------------------------------------------------------------------------
static inline void build_mode5_block(const uint8_t in_pix[16 * 4],
                                     uint8_t out_block[16]) {
    int r_min = 255, r_max = 0;
    int g_min = 255, g_max = 0;
    int b_min = 255, b_max = 0;
    int a_min = 255, a_max = 0;
    for (int i = 0; i < 16; i++) {
        int r = in_pix[i * 4 + 0];
        int g = in_pix[i * 4 + 1];
        int b = in_pix[i * 4 + 2];
        int a = in_pix[i * 4 + 3];
        if (r < r_min) r_min = r; if (r > r_max) r_max = r;
        if (g < g_min) g_min = g; if (g > g_max) g_max = g;
        if (b < b_min) b_min = b; if (b > b_max) b_max = b;
        if (a < a_min) a_min = a; if (a > a_max) a_max = a;
    }

    // SSE-optimized endpoints: min/max quantized to 7-bit (RGB) / 8-bit (A)
    uint32_t r0 = quant7(r_min), r1 = quant7(r_max);
    uint32_t g0 = quant7(g_min), g1 = quant7(g_max);
    uint32_t b0 = quant7(b_min), b1 = quant7(b_max);
    uint32_t a0 = (uint32_t)a_min, a1 = (uint32_t)a_max;

    uint32_t r0_8 = dq7(r0), r1_8 = dq7(r1);
    uint32_t g0_8 = dq7(g0), g1_8 = dq7(g1);
    uint32_t b0_8 = dq7(b0), b1_8 = dq7(b1);

    // Pick best 2-bit weight per pixel. Anchor (i=0) MSB is forced to 0,
    // so anchor weight index ∈ {0, 1} (WEIGHTS2[0]=0, WEIGHTS2[1]=21).
    uint32_t color_w[16], alpha_w[16];
    for (int i = 0; i < 16; i++) {
        int w_end = (i == 0) ? 2 : 4;
        // Color: minimize RGB SSE
        uint32_t bw_c = 0; int be_c = INT32_MAX;
        for (int wi = 0; wi < w_end; wi++) {
            uint32_t w = WEIGHTS2[wi];
            int dr = (int)interp(r0_8, r1_8, w) - in_pix[i * 4 + 0];
            int dg = (int)interp(g0_8, g1_8, w) - in_pix[i * 4 + 1];
            int db = (int)interp(b0_8, b1_8, w) - in_pix[i * 4 + 2];
            int e = dr * dr + dg * dg + db * db;
            if (e < be_c) { be_c = e; bw_c = (uint32_t)wi; }
        }
        color_w[i] = bw_c;
        // Alpha: minimize alpha SSE
        uint32_t bw_a = 0; int be_a = INT32_MAX;
        for (int wi = 0; wi < w_end; wi++) {
            uint32_t w = WEIGHTS2[wi];
            int da = (int)interp(a0, a1, w) - in_pix[i * 4 + 3];
            int e = da * da;
            if (e < be_a) { be_a = e; bw_a = (uint32_t)wi; }
        }
        alpha_w[i] = bw_a;
    }

    // Pack into Mode 5 bitstream (16 bytes, LSB-first).
    std::memset(out_block, 0, 16);
    out_block[0] = 0x20;  // mode 5: bit 5 set, comp_rot=0

    put_bits(out_block,  8, 7, r0);
    put_bits(out_block, 15, 7, r1);
    put_bits(out_block, 22, 7, g0);
    put_bits(out_block, 29, 7, g1);
    put_bits(out_block, 36, 7, b0);
    put_bits(out_block, 43, 7, b1);
    put_bits(out_block, 50, 8, a0);
    put_bits(out_block, 58, 8, a1);

    uint32_t bo = 66;
    put_bits(out_block, bo, 1, color_w[0] & 1);  bo += 1;  // anchor
    for (int i = 1; i < 16; i++) { put_bits(out_block, bo, 2, color_w[i]);  bo += 2; }
    put_bits(out_block, bo, 1, alpha_w[0] & 1);  bo += 1;  // anchor
    for (int i = 1; i < 16; i++) { put_bits(out_block, bo, 2, alpha_w[i]);  bo += 2; }
}

// ---------------------------------------------------------------------------
// Manual Mode 6 builder.
// 7-bit RGBA endpoints + 1 pbit per endpoint (shared across all 4 channels).
// 4-bit weights (16 levels) — much finer than Mode 5, so SSE is usually lower.
// Anchor at i=0 is 3-bit (MSB forced to 0, so anchor ∈ {0..7}).
//
// SSE-optimized: endpoints = min/max of input. Pbits searched (4 combinations)
// to minimize total SSE. Weights picked per-pixel to minimize RGBA SSE.
// ---------------------------------------------------------------------------
static inline void build_mode6_block(const uint8_t in_pix[16 * 4],
                                     uint8_t out_block[16]) {
    int r_min = 255, r_max = 0;
    int g_min = 255, g_max = 0;
    int b_min = 255, b_max = 0;
    int a_min = 255, a_max = 0;
    for (int i = 0; i < 16; i++) {
        int r = in_pix[i * 4 + 0];
        int g = in_pix[i * 4 + 1];
        int b = in_pix[i * 4 + 2];
        int a = in_pix[i * 4 + 3];
        if (r < r_min) r_min = r; if (r > r_max) r_max = r;
        if (g < g_min) g_min = g; if (g > g_max) g_max = g;
        if (b < b_min) b_min = b; if (b > b_max) b_max = b;
        if (a < a_min) a_min = a; if (a > a_max) a_max = a;
    }

    uint32_t r0 = quant7(r_min), r1 = quant7(r_max);
    uint32_t g0 = quant7(g_min), g1 = quant7(g_max);
    uint32_t b0 = quant7(b_min), b1 = quant7(b_max);
    uint32_t a0 = quant7(a_min), a1 = quant7(a_max);

    // Search over (p0, p1) ∈ {0,1}^2 — pick the combination with lowest SSE.
    uint32_t best_p0 = 0, best_p1 = 0;
    long best_sse = LONG_MAX;
    uint32_t best_weights[16] = {};
    for (int dp0 = 0; dp0 < 2; dp0++) {
        for (int dp1 = 0; dp1 < 2; dp1++) {
            uint32_t r0_8 = dq7p(r0, dp0), r1_8 = dq7p(r1, dp1);
            uint32_t g0_8 = dq7p(g0, dp0), g1_8 = dq7p(g1, dp1);
            uint32_t b0_8 = dq7p(b0, dp0), b1_8 = dq7p(b1, dp1);
            uint32_t a0_8 = dq7p(a0, dp0), a1_8 = dq7p(a1, dp1);

            long sse = 0;
            uint32_t weights[16];
            for (int i = 0; i < 16; i++) {
                int w_end = (i == 0) ? 8 : 16;
                uint32_t bw = 0; int be = INT32_MAX;
                for (int wi = 0; wi < w_end; wi++) {
                    uint32_t w = WEIGHTS4[wi];
                    int dr = (int)interp(r0_8, r1_8, w) - in_pix[i * 4 + 0];
                    int dg = (int)interp(g0_8, g1_8, w) - in_pix[i * 4 + 1];
                    int db = (int)interp(b0_8, b1_8, w) - in_pix[i * 4 + 2];
                    int da = (int)interp(a0_8, a1_8, w) - in_pix[i * 4 + 3];
                    int e = dr * dr + dg * dg + db * db + da * da;
                    if (e < be) { be = e; bw = (uint32_t)wi; }
                }
                weights[i] = bw;
                sse += be;
            }
            if (sse < best_sse) {
                best_sse = sse;
                best_p0 = (uint32_t)dp0;
                best_p1 = (uint32_t)dp1;
                std::memcpy(best_weights, weights, sizeof(weights));
            }
        }
    }
    uint32_t p0 = best_p0, p1 = best_p1;

    // Pack into Mode 6 bitstream (16 bytes, LSB-first).
    std::memset(out_block, 0, 16);
    out_block[0] = 0x40;  // mode 6: bit 6 set

    put_bits(out_block,  7, 7, r0);
    put_bits(out_block, 14, 7, r1);
    put_bits(out_block, 21, 7, g0);
    put_bits(out_block, 28, 7, g1);
    put_bits(out_block, 35, 7, b0);
    put_bits(out_block, 42, 7, b1);
    put_bits(out_block, 49, 7, a0);
    put_bits(out_block, 56, 7, a1);
    put_bits(out_block, 63, 1, p0);
    put_bits(out_block, 64, 1, p1);

    uint32_t bo = 65;
    put_bits(out_block, bo, 3, best_weights[0] & 0x7);  bo += 3;  // anchor (3-bit)
    for (int i = 1; i < 16; i++) {
        put_bits(out_block, bo, 4, best_weights[i]);
        bo += 4;
    }
}

// ---------------------------------------------------------------------------
// Env-var helpers (with sensible defaults).
// ---------------------------------------------------------------------------
static inline float envf(const char* name, float def) {
    if (const char* e = std::getenv(name)) {
        // Accept any explicit value including 0 ( atof("0") == 0.0 ).
        // Only fall back to def when the env var is absent.
        return (float)std::atof(e);
    }
    return def;
}

// --------------------------------------------------------------------
// Main entry point. NO DECODING — bias is computed exactly by reading
// the encoded bits (endpoints + pbits + partition + indexes) and applying
// the BC7 interpolation formula directly. The bc7decomp library is NOT
// invoked anywhere in this pass.
//
// Tuning (env vars, all optional):
//   M7_BIAS=f         trigger threshold (default 0.50; old default was 1.0)
//   M7_PERFECT_EPS=f  perfect-shape skip (default 0.20; never touch these)
//   M7_MIN_IMPROVE=f  required |bias| reduction fraction (default 0.25)
//   M7_SSE_TOL_PCT=f  SSE increase tolerance % (default 30)
//   M7_PERFECT_SSE=f  if sse_orig < this, no SSE increase allowed (default 50)
//   M7TO5=0           disable the fallback entirely
// --------------------------------------------------------------------
static int run(
    uint64_t* pBlocks,
    const uint32_t* pPixelsRGBA,
    uint32_t total_blocks,
    const bc7e_compress_block_params* base_params,
    float bias_threshold_arg = 1.0f,
    unsigned num_threads = 0)
{
    // ---- Resolve tuning knobs (env overrides > arg > default) ----
    // M7_BIAS env (if nonzero) overrides the caller-supplied bias_threshold_arg.
    float bias_threshold   = envf("M7_BIAS",        bias_threshold_arg);
    float perfect_eps      = envf("M7_PERFECT_EPS", 0.20f);
    float min_improve      = envf("M7_MIN_IMPROVE", 0.25f);
    float sse_tol_pct      = envf("M7_SSE_TOL_PCT", 10.0f);
    float perfect_sse_lim  = envf("M7_PERFECT_SSE", 100.0f);

    // M7TO5=0 disables the pass entirely.
    if (const char* e = std::getenv("M7TO5")) {
        if (std::atoi(e) == 0) {
            fprintf(stderr, "[M7TO5_NODECODE] DISABLED via M7TO5=0\n");
            return 0;
        }
    }

    if (num_threads == 0) {
        num_threads = std::thread::hardware_concurrency();
        if (num_threads == 0) num_threads = 4;
    }

    // Note: re-encoding uses the MANUAL M5/M6 builders (build_mode5_block /
    // build_mode6_block) below, NOT bc7e_compress_block_range. This bypasses
    // the encoder's max_color_range<128 and near_opaque guards that were
    // rejecting high-contrast candidates (the very blocks we need to fix).
    // base_params is preserved in the signature for API compatibility but
    // is not used by the manual builders.
    (void)base_params;

    std::atomic<int> applied{0};
    std::atomic<int> candidates{0};
    std::atomic<int> m5_wins{0};
    std::atomic<int> m6_wins{0};
    std::atomic<int> skip_perfect{0};      // |bias| < perfect_eps
    std::atomic<int> skip_below_trigger{0}; // perfect_eps <= |bias| < trigger
    std::atomic<int> reject_no_improve{0};  // candidate fails MIN_IMPROVE (M5+M6 combined)
    std::atomic<int> reject_sse{0};         // candidate fails SSE guard (M5+M6 combined)
    std::atomic<int> m5_reject_bias{0};
    std::atomic<int> m5_reject_sse{0};
    std::atomic<int> m6_reject_bias{0};
    std::atomic<int> m6_reject_sse{0};
    std::atomic<long> mode_src_count[8]{};
    std::atomic<uint32_t> next{0};
    const uint32_t chunk = 64;

    std::vector<std::thread> ths;
    ths.reserve(num_threads);
    for (unsigned t = 0; t < num_threads; t++) {
        ths.emplace_back([&]() {
            uint8_t in_pix[16 * 4];
            uint64_t m5_block[2];
            uint64_t m6_block[2];

            for (;;) {
                uint32_t b0 = next.fetch_add(chunk, std::memory_order_relaxed);
                if (b0 >= total_blocks) break;
                uint32_t b1 = std::min(b0 + chunk, total_blocks);

                for (uint32_t bi = b0; bi < b1; bi++) {
                    uint64_t* pBlk = &pBlocks[bi * 2];
                    int mode = detect_mode(pBlk);
                    if (mode < 0) continue;
                    // Selective: only Mode 4 (5-bit, 1-subset) and Mode 7
                    // (5-bit, 2-subset) — the bias-prone modes.
                    if (mode != 4 && mode != 7) continue;

                    const uint8_t* in_src = (const uint8_t*)&pPixelsRGBA[bi * 16];
                    memcpy(in_pix, in_src, 16 * 4);

                    // Compute EXACT bias from encoded bits — NO DECODER CALL.
                    double bias_orig = (mode == 4)
                        ? mode4_green_bias((uint8_t*)pBlk, in_pix)
                        : mode7_green_bias((uint8_t*)pBlk, in_pix);
                    double abs_b_orig = std::fabs(bias_orig);

                    // GUARD 1: PERFECT SHAPE — never touch already-good blocks.
                    // This is what prevents the bottom of a glowing up-arrow
                    // from getting slightly more blocky on specific spots.
                    if (abs_b_orig < perfect_eps) {
                        skip_perfect.fetch_add(1, std::memory_order_relaxed);
                        continue;
                    }

                    // GUARD 2: TRIGGER — must be bad enough to consider fixing.
                    if (abs_b_orig < bias_threshold) {
                        skip_below_trigger.fetch_add(1, std::memory_order_relaxed);
                        continue;
                    }

                    candidates.fetch_add(1, std::memory_order_relaxed);
                    mode_src_count[mode].fetch_add(1, std::memory_order_relaxed);

                    // Compute SSE of original (used by SSE guard).
                    double sse_orig = (mode == 4)
                        ? mode4_rgb_sse((uint8_t*)pBlk, in_pix)
                        : mode7_rgb_sse((uint8_t*)pBlk, in_pix);

                    // Determine SSE limit for candidates.
                    bool orig_perfect_sse = (sse_orig < perfect_sse_lim);
                    double sse_limit = orig_perfect_sse
                        ? sse_orig                          // no increase allowed
                        : sse_orig * (1.0 + sse_tol_pct / 100.0);

                    // Required bias improvement: candidate |bias| must be
                    // <= (1 - min_improve) * |bias_orig|.
                    double bias_target = abs_b_orig * (1.0 - min_improve);

                    // Build Mode 5 candidate manually (SSE-optimized min/max endpoints,
                    // 2-bit weights per pixel).
                    build_mode5_block(in_pix, (uint8_t*)m5_block);
                    bool m5_valid = true;  // manual builder always produces M5

                    double bias_m5 = mode5_green_bias((uint8_t*)m5_block, in_pix);
                    double sse_m5  = mode5_rgb_sse((uint8_t*)m5_block, in_pix);
                    if (std::fabs(bias_m5) > bias_target) {
                        m5_valid = false;
                        m5_reject_bias.fetch_add(1, std::memory_order_relaxed);
                    } else if (sse_m5 > sse_limit) {
                        m5_valid = false;
                        m5_reject_sse.fetch_add(1, std::memory_order_relaxed);
                    }

                    // Build Mode 6 candidate manually (7-bit + pbit endpoints,
                    // 4-bit weights — much finer granularity than M5).
                    build_mode6_block(in_pix, (uint8_t*)m6_block);
                    bool m6_valid = true;  // manual builder always produces M6

                    double bias_m6 = mode6_green_bias((uint8_t*)m6_block, in_pix);
                    double sse_m6  = mode6_rgb_sse((uint8_t*)m6_block, in_pix);
                    if (std::fabs(bias_m6) > bias_target) {
                        m6_valid = false;
                        m6_reject_bias.fetch_add(1, std::memory_order_relaxed);
                    } else if (sse_m6 > sse_limit) {
                        m6_valid = false;
                        m6_reject_sse.fetch_add(1, std::memory_order_relaxed);
                    }

                    // Pick the candidate. Tiebreak by lower SSE (which also
                    // rejects brightening — a candidate that brightens will
                    // have higher SSE than one that doesn't).
                    const uint64_t* best_block = nullptr;
                    int best_mode = -1;
                    if (m5_valid && m6_valid) {
                        if (sse_m5 <= sse_m6) {
                            best_block = m5_block;
                            best_mode = 5;
                        } else {
                            best_block = m6_block;
                            best_mode = 6;
                        }
                    } else if (m5_valid) {
                        best_block = m5_block;
                        best_mode = 5;
                    } else if (m6_valid) {
                        best_block = m6_block;
                        best_mode = 6;
                    }

                    if (best_block) {
                        pBlk[0] = best_block[0];
                        pBlk[1] = best_block[1];
                        applied.fetch_add(1, std::memory_order_relaxed);
                        if (best_mode == 5) m5_wins.fetch_add(1, std::memory_order_relaxed);
                        else                m6_wins.fetch_add(1, std::memory_order_relaxed);
                    }
                }
            }
        });
    }
    for (auto& th : ths) th.join();

    fprintf(stderr, "[M7TO5_NODECODE] candidates=%d  applied=%d  (m5_wins=%d  m6_wins=%d)\n",
            candidates.load(), applied.load(), m5_wins.load(), m6_wins.load());
    fprintf(stderr, "  skipped: perfect=%d  below_trigger=%d\n",
            skip_perfect.load(), skip_below_trigger.load());
    fprintf(stderr, "  M5: reject_bias=%d  reject_sse=%d | M6: reject_bias=%d  reject_sse=%d\n",
            m5_reject_bias.load(), m5_reject_sse.load(),
            m6_reject_bias.load(), m6_reject_sse.load());
    fprintf(stderr, "  Source modes:");
    for (int m = 0; m < 8; m++) {
        long c = mode_src_count[m].load();
        if (c > 0) fprintf(stderr, "  M%d=%ld", m, c);
    }
    fprintf(stderr, "\n");
    fprintf(stderr, "  [trigger=%.2f  perfect_eps=%.2f  min_improve=%.2f  sse_tol_pct=%.1f  perfect_sse=%.0f]\n",
            bias_threshold, perfect_eps, min_improve, sse_tol_pct, perfect_sse_lim);

    return applied.load();
}

} // namespace m7to5_nodecode
