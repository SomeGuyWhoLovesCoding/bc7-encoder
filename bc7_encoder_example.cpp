// bc7_encoder_example.cpp - Multithreaded scalar BC7 encoder with atomic work stealing
//
// IMPORTANT: This must be compiled as a single translation unit because
// bc7_encoder_base.h defines non-inline functions and globals (g_codec_initialized).
//
// SIMD paths (bc7_encoder_simd.h / bc7_encoder_dispatch.h / bc7_encoder_simd_impl.inc)
// have been stripped entirely. The encoder uses the scalar path
// (bc7e_compress_block_range from bc7_encoder_base.h) only.
//
// Rationale: on 3950x1680 input the AVX2 8-wide path was hard-wired to
// mode 6 (Paeth) only and was both broken AND slower than scalar. SSE2/SSE4
// 4-wide paths were also slower than scalar in practice because the
// dispatch + per-block setup overhead exceeded the per-block SIMD gain.
// Scalar-only is simpler AND faster, so we keep only that path.
//
// Build (GCC/Clang) - no -mavx2 needed:
//   g++ -std=c++17 -O2 -pthread -I. -o bc7_encoder_example bc7_encoder_example.cpp -lm
//
// Build (MSVC):
//   cl /EHsc /O2 /std:c++17 /I. bc7_encoder_example.cpp /Fe:bc7_encoder_example.exe
//
// Usage:
//   bc7_encoder_example input.png output.dds
//
// Quality is locked to level 2 (default/balanced).
// Uses all available CPU cores via std::thread with atomic work stealing 
// (chunked dynamic scheduling) to ensure optimal load balancing and prevent 
// tail latency from variable block compression times.

#include "bc7_encoder_base.h"
#include "bc7_metrics.h"

// NOTE: bc7_encoder_simd.h, bc7_encoder_dispatch.h, and
// bc7_encoder_simd_impl.inc are intentionally NOT included.
// We call bc7e_compress_block_range (scalar) from bc7_encoder_base.h directly.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <thread>
#include <chrono>
#include <algorithm>
#include <atomic>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

//
// Minimal DDS writer (BC7, DX10 extended header)
//
#pragma pack(push, 1)
struct DDS_PIXELFORMAT {
    uint32_t size;
    uint32_t flags;
    uint32_t fourCC;
    uint32_t rgbBitCount;
    uint32_t rBitMask;
    uint32_t gBitMask;
    uint32_t bBitMask;
    uint32_t aBitMask;
};

struct DDS_HEADER {
    uint32_t size;
    uint32_t flags;
    uint32_t height;
    uint32_t width;
    uint32_t pitchOrLinearSize;
    uint32_t depth;
    uint32_t mipMapCount;
    uint32_t reserved1[11];
    DDS_PIXELFORMAT ddspf;
    uint32_t caps;
    uint32_t caps2;
    uint32_t caps3;
    uint32_t caps4;
    uint32_t reserved2;
};

struct DDS_HEADER_DXT10 {
    uint32_t dxgiFormat;
    uint32_t resourceDimension;
    uint32_t miscFlag;
    uint32_t arraySize;
    uint32_t miscFlags2;
};
#pragma pack(pop)

static bool write_dds_bc7(const char* filename, int width, int height,
                          const uint8_t* bc7_data, size_t bc7_size)
{
    FILE* f = fopen(filename, "wb");
    if (!f) {
        fprintf(stderr, "Error: Cannot open output file '%s'\n", filename);
        return false;
    }

    fwrite("DDS ", 1, 4, f);

    DDS_HEADER hdr = {};
    hdr.size = sizeof(DDS_HEADER);
    hdr.flags = 0x82008;
    hdr.height = (uint32_t)height;
    hdr.width = (uint32_t)width;
    hdr.pitchOrLinearSize = (uint32_t)bc7_size;
    hdr.depth = 1;
    hdr.mipMapCount = 1;
    hdr.ddspf.size = sizeof(DDS_PIXELFORMAT);
    hdr.ddspf.flags = 0x4;
    hdr.ddspf.fourCC = 0x30315844; // 'DX10'
    hdr.caps = 0x1000;
    fwrite(&hdr, sizeof(DDS_HEADER), 1, f);

    DDS_HEADER_DXT10 dx10 = {};
    dx10.dxgiFormat = 98; // DXGI_FORMAT_BC7_UNORM
    dx10.resourceDimension = 3;
    dx10.arraySize = 1;
    fwrite(&dx10, sizeof(DDS_HEADER_DXT10), 1, f);

    fwrite(bc7_data, 1, bc7_size, f);
    fclose(f);
    return true;
}

class BC7Encoder {
public:
    // Function pointer type matching bc7e_compress_block_range's signature.
    using compress_fn = void (*)(uint32_t first_block, uint32_t count,
                                 uint64_t* pBlocks, const uint32_t* pPixelsRGBA,
                                 const bc7e_compress_block_params* pComp_params);

    BC7Encoder() = default;

    static const char* get_encoder_path_string() {
        return "Scalar (Atomic Work Stealing)";
    }

    void encode_image_mt(const uint8_t* rgba, int w, int h, std::vector<uint8_t>& bc7_out) {
        const int BW = 4, BH = 4;
        int bx_count = (w + BW - 1) / BW;
        int by_count = (h + BH - 1) / BH;
        uint32_t total_blocks = (uint32_t)(bx_count * by_count);

        // Prepare input pixels: pad to multiple of 4x4, pack as uint32_t RGBA
        int pw = bx_count * BW;
        int ph = by_count * BH;
        std::vector<uint32_t> pixels_rgba(total_blocks * 16, 0);

        for (int y = 0; y < ph; y++) {
            int sy = (y < h) ? y : (h - 1);
            for (int x = 0; x < pw; x++) {
                int sx = (x < w) ? x : (w - 1);
                int si = (sy * w + sx) * 4;
                uint32_t pixel = (uint32_t)rgba[si + 0]
                              | ((uint32_t)rgba[si + 1] << 8)
                              | ((uint32_t)rgba[si + 2] << 16)
                              | ((uint32_t)rgba[si + 3] << 24);
                int block_idx = (y / BH) * bx_count + (x / BW);
                int pixel_in_block = (y % BH) * BW + (x % BW);
                pixels_rgba[block_idx * 16 + pixel_in_block] = pixel;
            }
        }

        // Quality 2 (default/balanced)
        bc7e_compress_block_params params;
        bc7e_compress_block_params_init(&params, false);
        params.m_pbit_search = false;
        params.m_refinement_passes = 2;
        //params.m_uber_level = 1;   // enables the PCA seed improvement (mode 1/3)

        // bc7e_compress_block_init() initializes the codec's lookup tables
        // (defined in bc7_encoder_base.h, independent of any SIMD path).
        bc7e_compress_block_init();

        // Determine thread count: use hardware concurrency, capped at block count
        unsigned num_threads = std::min(
            std::max(1u, std::thread::hardware_concurrency()),
            (unsigned)total_blocks);

        std::vector<uint64_t> blocks(total_blocks * 2);

        // ---- TIMED ENCODE ----
        auto t0 = std::chrono::high_resolution_clock::now();

        if (num_threads <= 1) {
            // Single-threaded scalar encode
            g_encode_fn(0, total_blocks, blocks.data(), pixels_rgba.data(), &params);
        } else {
            // Multi-threaded scalar encode with atomic work stealing (chunked dynamic scheduling).
            // Instead of statically partitioning blocks, threads atomically fetch chunks of work.
            // This prevents tail latency caused by variable block compression times.
            std::atomic<uint32_t> next_block{0};
            
            // Chunk size balances atomic contention overhead vs load balancing granularity.
            // 64 blocks per chunk is a robust default for BC7 scalar encoding. 
            // Tune this value (e.g., 32, 64, 128) based on your specific CPU and image characteristics.
            constexpr uint32_t chunk_size = 64;

            auto worker = [&]() {
                while (true) {
                    // Atomically claim a chunk of blocks
                    uint32_t start = next_block.fetch_add(chunk_size, std::memory_order_relaxed);
                    if (start >= total_blocks) {
                        break; // No more work to steal
                    }
                    
                    uint32_t count = std::min(chunk_size, total_blocks - start);
                    g_encode_fn(start, count, blocks.data(), pixels_rgba.data(), &params);
                }
            };

            std::vector<std::thread> threads;
            threads.reserve(num_threads);
            for (unsigned t = 0; t < num_threads; ++t) {
                threads.emplace_back(worker);
            }

            for (auto& th : threads) {
                th.join();
            }
        }

        auto t1 = std::chrono::high_resolution_clock::now();
        double encode_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        fprintf(stderr, "Encoded %u blocks in %.1f ms (%.0f blocks/sec, %u threads, %s)\n",
                total_blocks, encode_ms,
                encode_ms > 0 ? (total_blocks / (encode_ms / 1000.0)) : 0.0,
                num_threads, get_encoder_path_string());

        bc7_out.resize(total_blocks * 16);
        memcpy(bc7_out.data(), blocks.data(), total_blocks * 16);
    }

private:
    // Hard-point the function pointer at the scalar implementation from base.h.
    // No CPUID detection, no SIMD fallback ladder, no per-thread dispatch cache.
    static const compress_fn g_encode_fn;
};

// Initialize the static function pointer
const BC7Encoder::compress_fn BC7Encoder::g_encode_fn = bc7e_compress_block_range;

static void generate_mode_visualization(const char* filename, const uint8_t* bc7_data, int width, int height) {
    int bw = (width + 3) / 4;
    int bh = (height + 3) / 4;
    
    // Colors for each mode (RGB)
    uint8_t mode_colors[8][3] = {
        {0, 255, 255},   // Mode 0: Cyan
        {255, 0, 0},     // Mode 1: Red
        {0, 255, 0},     // Mode 2: Green
        {255, 255, 0},   // Mode 3: Yellow
        {255, 0, 255},   // Mode 4: Magenta
        {255, 128, 0},   // Mode 5: Orange
        {0, 0, 255},     // Mode 6: Blue
        {255, 255, 255}  // Mode 7: White
    };

    std::vector<uint8_t> vis(width * height * 3, 0);

    for (int by = 0; by < bh; by++) {
        for (int bx = 0; bx < bw; bx++) {
            int block_idx = by * bw + bx;
            uint8_t first_byte = bc7_data[block_idx * 16];
            
            // Extract mode from the first byte (lowest set bit)
            uint32_t mode = 0;
            while (mode < 8 && !(first_byte & (1 << mode))) mode++;
            if (mode >= 8) mode = 7; // fallback

            uint8_t r = mode_colors[mode][0];
            uint8_t g = mode_colors[mode][1];
            uint8_t b = mode_colors[mode][2];

            for (int py = 0; py < 4; py++) {
                for (int px = 0; px < 4; px++) {
                    int x = bx * 4 + px;
                    int y = by * 4 + py;
                    if (x < width && y < height) {
                        int di = (y * width + x) * 3;
                        vis[di + 0] = r;
                        vis[di + 1] = g;
                        vis[di + 2] = b;
                    }
                }
            }
        }
    }

    // Write as PPM (P6) - universally supported, no extra libraries needed
    FILE* f = fopen(filename, "wb");
    if (f) {
        fprintf(f, "P6\n%d %d\n255\n", width, height);
        fwrite(vis.data(), 1, vis.size(), f);
        fclose(f);
        fprintf(stderr, "Wrote mode visualization to %s\n", filename);
    } else {
        fprintf(stderr, "Error: Could not write mode visualization to %s\n", filename);
    }
}

int main(int argc, char* argv[])
{
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <input.png> <output.dds>\n", argv[0]);
        return 1;
    }

    const char* in_path = argv[1];
    const char* out_path = argv[2];

    int w = 0, h = 0, ch = 0;
    uint8_t* pixels = stbi_load(in_path, &w, &h, &ch, 4);
    if (!pixels) {
        fprintf(stderr, "Error: Failed to load '%s': %s\n", in_path, stbi_failure_reason());
        return 1;
    }

    fprintf(stderr, "Loaded %dx%d image \"%s\" - %d channels\n", w, h, in_path, ch);

    BC7Encoder encoder;
    std::vector<uint8_t> bc7;
    encoder.encode_image_mt(pixels, w, h, bc7);

    size_t orig = (size_t)w * h * 4;
    fprintf(stderr, "Original: %zu bytes, BC7: %zu bytes (%.2f:1, %.1f%%)\n",
            orig, bc7.size(), (double)orig / (double)bc7.size(),
            100.0 * (double)bc7.size() / (double)orig);

    if (!write_dds_bc7(out_path, w, h, bc7.data(), bc7.size())) {
        stbi_image_free(pixels);
        return 1;
    }
    fprintf(stderr, "Wrote %s\n", out_path);

    // ── Mode Visualization ──
    // Generates a PPM image where each 4x4 block is colored by its chosen BC7 mode.
    std::string mode_vis_path = out_path;
    size_t dot_pos = mode_vis_path.find_last_of('.');
    if (dot_pos != std::string::npos) {
        mode_vis_path = mode_vis_path.substr(0, dot_pos) + "_modes.ppm";
    } else {
        mode_vis_path += "_modes.ppm";
    }
    generate_mode_visualization(mode_vis_path.c_str(), bc7.data(), w, h);

    // ── Quality Metrics (matches bc7enc output format) ──
    compute_and_print_all_metrics(pixels, bc7.data(), bc7.size(), w, h);

    stbi_image_free(pixels);
    return 0;
}