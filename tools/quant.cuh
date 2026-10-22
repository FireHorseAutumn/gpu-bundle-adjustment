#pragma once

// Quantize/dequantize roundtrip helpers for the FP8/FP4 feasibility study.
// Uses the CUDA FP8/FP4 conversion intrinsics (host path) so the grids are
// hardware-exact, not hand-rolled approximations.
//
// Ground rule from an archived-pipeline failure: overflow and NaN are
// counted explicitly, never
// silently clamped away. Every call site can pass a Stats* and must report it.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>

#include <cmath>

namespace gba::quant {

enum class Precision { FP32, FP16, BF16, E4M3, E5M2, FP4 };

inline const char* name(Precision p) {
    switch (p) {
        case Precision::FP32: return "fp32";
        case Precision::FP16: return "fp16";
        case Precision::BF16: return "bf16";
        case Precision::E4M3: return "e4m3";
        case Precision::E5M2: return "e5m2";
        case Precision::FP4: return "fp4e2m1";
    }
    return "?";
}

inline double max_representable(Precision p) {
    switch (p) {
        case Precision::FP32: return 3.402823466e38;
        case Precision::FP16: return 65504.0;
        case Precision::BF16: return 3.38953139e38;
        case Precision::E4M3: return 448.0;
        case Precision::E5M2: return 57344.0;
        case Precision::FP4: return 6.0;
    }
    return 0.0;
}

// Smallest positive representable (subnormal floor).
inline double min_representable(Precision p) {
    switch (p) {
        case Precision::FP32: return 1e-45;
        case Precision::FP16: return 5.9604645e-08;  // 2^-24
        case Precision::BF16: return 9.1835496e-41;  // 2^-133
        case Precision::E4M3: return 0.001953125;   // 2^-9
        case Precision::E5M2: return 1.5258789e-05; // 2^-16
        case Precision::FP4: return 0.5;
    }
    return 0.0;
}

struct Stats {
    long overflow = 0;   // |v| > max representable (saturated)
    long underflow = 0;  // 0 < |v| < min representable (flushed toward zero)
    long nan = 0;
};

// Quantize v onto the grid of p, convert back to double.
// FP32 passes through a float cast (no scale semantics).
inline double roundtrip(Precision p, double v, Stats* st = nullptr) {
    if (p == Precision::FP32) return static_cast<double>(static_cast<float>(v));
    if (p == Precision::FP16)
        return static_cast<double>(static_cast<float>(__half(static_cast<float>(v))));
    if (p == Precision::BF16)
        return static_cast<double>(static_cast<float>(__nv_bfloat16(static_cast<float>(v))));
    if (std::isnan(v)) {
        if (st) ++st->nan;
        return v;
    }
    const double av = std::fabs(v);
    if (av > max_representable(p)) {
        if (st) ++st->overflow;
        v = std::copysign(max_representable(p), v);
    } else if (av > 0.0 && av < min_representable(p)) {
        if (st) ++st->underflow;
    }
    __half_raw hr;
    switch (p) {
        case Precision::E4M3:
            hr = __nv_cvt_fp8_to_halfraw(
                __nv_cvt_double_to_fp8(v, __NV_SATFINITE, __NV_E4M3), __NV_E4M3);
            break;
        case Precision::E5M2:
            hr = __nv_cvt_fp8_to_halfraw(
                __nv_cvt_double_to_fp8(v, __NV_SATFINITE, __NV_E5M2), __NV_E5M2);
            break;
        case Precision::FP4:
            hr = __nv_cvt_fp4_to_halfraw(
                __nv_cvt_double_to_fp4(v, __NV_E2M1, cudaRoundNearest), __NV_E2M1);
            break;
        default:
            return v;
    }
    return static_cast<double>(static_cast<float>(__half(hr)));
}

}  // namespace gba::quant
