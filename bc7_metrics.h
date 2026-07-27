// bc7_metrics.h - Quality metrics matching bc7enc's exact implementation
// dont need this no more just 
#pragma once

#include <cstdint>
#include <cstdio>
#include <cmath>
#include <vector>
#include <algorithm>

// Use the official bc7decomp decoder from bc7enc
#define BC7DECOMP_IMPLEMENTATION
#include "bc7decomp.cpp"
#include "bc7decomp.h"

static inline int metrics_iabs(int i) { return i < 0 ? -i : i; }

// Exact replica of bc7enc's image_metrics::compute logic
static void compute_and_print_all_metrics(const uint8_t* orig_rgba,
                                          const uint8_t* bc7_data, size_t bc7_size,
                                          int width, int height)
{
    int bw = (width + 3) / 4;
    int bh = (height + 3) / 4;
    int num_pixels = width * height;

    std::vector<uint8_t> decoded(width * height * 4, 0);

    // Decode using the official bc7decomp
    for (int by = 0; by < bh; by++) {
        for (int bx = 0; bx < bw; bx++) {
            const uint8_t* blk = bc7_data + (by * bw + bx) * 16;
            bc7decomp::color_rgba pixels[16];
            bc7decomp::unpack_bc7(blk, pixels);

            for (int py = 0; py < 4; py++) {
                for (int px = 0; px < 4; px++) {
                    int x = bx * 4 + px;
                    int y = by * 4 + py;
                    if (x < width && y < height) {
                        int di = (y * width + x) * 4;
                        int si = py * 4 + px;
                        decoded[di + 0] = pixels[si].r;
                        decoded[di + 1] = pixels[si].g;
                        decoded[di + 2] = pixels[si].b;
                        decoded[di + 3] = pixels[si].a;
                    }
                }
            }
        }
    }

    // --- Luma (BT.709 weights exactly like bc7enc) ---
    {
        double hist[256] = {};
        for (int i = 0; i < num_pixels; i++) {
            int ya = (13938U * orig_rgba[i*4+0] + 46869U * orig_rgba[i*4+1] + 4729U * orig_rgba[i*4+2] + 32768U) >> 16U;
            int yb = (13938U * decoded[i*4+0] + 46869U * decoded[i*4+1] + 4729U * decoded[i*4+2] + 32768U) >> 16U;
            hist[metrics_iabs(ya - yb)]++;
        }
        double max_e = 0, sum = 0, sum2 = 0;
        for (int i = 0; i < 256; i++) {
            if (!hist[i]) continue;
            max_e = std::max(max_e, (double)i);
            double x = i * hist[i];
            sum += x;
            sum2 += i * x;
        }
        double mse = sum2 / num_pixels;
        double rmse = sqrt(mse);
        double psnr = rmse > 0.0 ? log10(255.0 / rmse) * 20.0 : 100.0;
        printf("Luma  Max error: %3.0f RMSE: %f PSNR %5.2f dB\n", max_e, rmse, psnr);
    }

    // --- RGB ---
    {
        double hist[256] = {};
        for (int i = 0; i < num_pixels; i++) {
            for (int c = 0; c < 3; c++) {
                hist[metrics_iabs((int)orig_rgba[i*4+c] - (int)decoded[i*4+c])]++;
            }
        }
        double max_e = 0, sum = 0, sum2 = 0;
        for (int i = 0; i < 256; i++) {
            if (!hist[i]) continue;
            max_e = std::max(max_e, (double)i);
            double x = i * hist[i];
            sum += x;
            sum2 += i * x;
        }
        double total = num_pixels * 3.0; // Averaged over 3 channels
        double mse = sum2 / total;
        double rmse = sqrt(mse);
        double psnr = rmse > 0.0 ? log10(255.0 / rmse) * 20.0 : 100.0;
        printf("RGB   Max error: %3.0f RMSE: %f PSNR %5.2f dB\n", max_e, rmse, psnr);
    }

    // --- RGBA ---
    {
        double hist[256] = {};
        for (int i = 0; i < num_pixels; i++) {
            for (int c = 0; c < 4; c++) {
                hist[metrics_iabs((int)orig_rgba[i*4+c] - (int)decoded[i*4+c])]++;
            }
        }
        double max_e = 0, sum = 0, sum2 = 0;
        for (int i = 0; i < 256; i++) {
            if (!hist[i]) continue;
            max_e = std::max(max_e, (double)i);
            double x = i * hist[i];
            sum += x;
            sum2 += i * x;
        }
        double total = num_pixels * 4.0; // Averaged over 4 channels
        double mse = sum2 / total;
        double rmse = sqrt(mse);
        double psnr = rmse > 0.0 ? log10(255.0 / rmse) * 20.0 : 100.0;
        printf("RGBA  Max error: %3.0f RMSE: %f PSNR %5.2f dB\n", max_e, rmse, psnr);
    }

    // --- Individual Channels ---
    const char* names[4] = { "Red  ", "Green", "Blue ", "Alpha" };
    for (int c = 0; c < 4; c++) {
        double hist[256] = {};
        for (int i = 0; i < num_pixels; i++) {
            hist[metrics_iabs((int)orig_rgba[i*4+c] - (int)decoded[i*4+c])]++;
        }
        double max_e = 0, sum = 0, sum2 = 0;
        for (int i = 0; i < 256; i++) {
            if (!hist[i]) continue;
            max_e = std::max(max_e, (double)i);
            double x = i * hist[i];
            sum += x;
            sum2 += i * x;
        }
        double mse = sum2 / num_pixels;
        double rmse = sqrt(mse);
        double psnr = rmse > 0.0 ? log10(255.0 / rmse) * 20.0 : 100.0;
        printf("%s Max error: %3.0f RMSE: %f PSNR %5.2f dB\n", names[c], max_e, rmse, psnr);
    }
}