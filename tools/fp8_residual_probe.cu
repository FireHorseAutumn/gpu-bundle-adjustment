// fp8_residual_probe — evaluate the Snavely reprojection residual under
// low precision, per observation, against the double reference. This tests
// ARITHMETIC quantization: what a kernel that computes in FP8/FP4 would
// produce, before any kernel is written.
//
// Modes:
//   storage — quantize inputs (camera9/point3/observed2), compute in double.
//             Matches the Phase-2 kernel design: FP8 storage, FP32+ compute.
//   arith   — quantize inputs AND round every arithmetic op to the grid.
//             The aggressive bound: full-FP8 math.
//
// Usage:
//   fp8_residual_probe <problem.txt[.bz2]> --precision=e4m3|e5m2|fp4
//                      --mode=storage|arith [--scale=none|auto|VALUE]
//
// Output: JSON — input/intermediate overflow & NaN counts (explicit, per
// the archived-pipeline failure rule), residual error distribution in pixels
// vs the double
// reference, and garbage-threshold counts.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "gba/bal_problem.hpp"
#include "quant.cuh"

namespace {

using gba::quant::Precision;
using gba::quant::Stats;

Precision g_prec = Precision::E4M3;
Stats g_stats;  // counts input + intermediate saturation/NaN events

double q(double v) { return gba::quant::roundtrip(g_prec, v, &g_stats); }

// Q: double value that re-quantizes after every op (arith mode).
struct Q {
    double v;
};
Q qr(double v) { return Q{q(v)}; }
Q operator+(Q a, Q b) { return qr(a.v + b.v); }
Q operator-(Q a, Q b) { return qr(a.v - b.v); }
Q operator-(Q a) { return qr(-a.v); }
Q operator*(Q a, Q b) { return qr(a.v * b.v); }
Q operator/(Q a, Q b) { return qr(a.v / b.v); }
Q qsqrt(Q a) { return qr(std::sqrt(a.v)); }
Q qcos(Q a) { return qr(std::cos(a.v)); }
Q qsin(Q a) { return qr(std::sin(a.v)); }

// snavely_residual with Q arithmetic, same structure as bal_reader.cpp.
void residual_arith(const double* camera9, const double* point3,
                    const double* observed2, double s_cam, double s_pts,
                    double s_pix, double* out2) {
    Q rod[3], t[3];
    for (int i = 0; i < 3; ++i) {
        rod[i] = qr(q(camera9[i] * s_cam) / s_cam);
        t[i] = qr(q(camera9[3 + i] * s_cam) / s_cam);
    }
    Q f = qr(q(camera9[6] * s_cam) / s_cam);
    Q k1 = qr(q(camera9[7] * s_cam) / s_cam);
    Q k2 = qr(q(camera9[8] * s_cam) / s_cam);
    Q X[3];
    for (int i = 0; i < 3; ++i) X[i] = qr(q(point3[i] * s_pts) / s_pts);

    Q theta = qsqrt(rod[0] * rod[0] + rod[1] * rod[1] + rod[2] * rod[2]);
    Q P[3];
    if (theta.v > 1e-12) {
        Q w[3] = {rod[0] / theta, rod[1] / theta, rod[2] / theta};
        Q c = qcos(theta), s = qsin(theta);
        Q wx = w[0] * X[0] + w[1] * X[1] + w[2] * X[2];
        Q cross[3] = {w[1] * X[2] - w[2] * X[1], w[2] * X[0] - w[0] * X[2],
                      w[0] * X[1] - w[1] * X[0]};
        Q one_c = qr(1.0) - c;
        for (int i = 0; i < 3; ++i)
            P[i] = X[i] * c + cross[i] * s + w[i] * wx * one_c + t[i];
    } else {
        for (int i = 0; i < 3; ++i) P[i] = X[i] + t[i];
    }

    Q px = -P[0] / P[2], py = -P[1] / P[2];
    Q r2 = px * px + py * py;
    Q r = qr(1.0) + k1 * r2 + k2 * r2 * r2;
    Q obs0 = qr(q(observed2[0] * s_pix) / s_pix);
    Q obs1 = qr(q(observed2[1] * s_pix) / s_pix);
    out2[0] = (f * r * px - obs0).v;
    out2[1] = (f * r * py - obs1).v;
}

void residual_storage(const double* camera9, const double* point3,
                      const double* observed2, double s_cam, double s_pts,
                      double s_pix, double* out2) {
    double c[9], X[3], o[2];
    for (int i = 0; i < 9; ++i) c[i] = q(camera9[i] * s_cam) / s_cam;
    for (int i = 0; i < 3; ++i) X[i] = q(point3[i] * s_pts) / s_pts;
    for (int i = 0; i < 2; ++i) o[i] = q(observed2[i] * s_pix) / s_pix;
    gba::snavely_residual(c, X, o, out2);
}

double maxabs(const std::vector<double>& v) {
    double m = 0.0;
    for (double x : v) m = std::max(m, std::fabs(x));
    return m;
}

}  // namespace

int main(int argc, char** argv) {
    std::string file, scale = "none", mode;
    for (int i = 1; i < argc; ++i) {
        std::string s = argv[i];
        auto eat = [&](const char* key) -> std::string {
            return s.rfind(key, 0) == 0 ? s.substr(std::strlen(key)) : "";
        };
        if (std::string v = eat("--precision="); !v.empty()) {
            if (v == "e4m3") g_prec = Precision::E4M3;
            else if (v == "e5m2") g_prec = Precision::E5M2;
            else if (v == "fp4") g_prec = Precision::FP4;
            else { std::fprintf(stderr, "bad precision (arith probe is fp8/fp4 only)\n"); return 1; }
        } else if (std::string v = eat("--mode="); !v.empty()) {
            mode = v;
        } else if (std::string v = eat("--scale="); !v.empty()) {
            scale = v;
        } else if (s.rfind("--", 0) != 0 && file.empty()) {
            file = s;
        } else {
            std::fprintf(stderr, "bad argument: %s\n", s.c_str());
            return 1;
        }
    }
    if (file.empty() || (mode != "storage" && mode != "arith")) {
        std::fprintf(stderr,
                     "usage: fp8_residual_probe <problem> --precision=e4m3|e5m2|fp4 "
                     "--mode=storage|arith [--scale=none|auto|VALUE]\n");
        return 1;
    }

    gba::BALProblem bal = gba::load_bal(file);

    double s_cam = 1.0, s_pts = 1.0, s_pix = 1.0;
    if (scale == "auto") {
        const double m = 0.9 * gba::quant::max_representable(g_prec);
        s_cam = m / std::max(1e-12, maxabs(bal.cameras));
        s_pts = m / std::max(1e-12, maxabs(bal.points));
        s_pix = m / std::max(1e-12, maxabs(bal.obs_pixel));
    } else if (scale != "none") {
        s_cam = s_pts = s_pix = std::stod(scale);
    }

    std::vector<double> errs;
    errs.reserve(bal.num_observations);
    double ref_cost2 = 0.0;  // 2 * reference cost
    long ref_nan = 0;
    for (int i = 0; i < bal.num_observations; ++i) {
        const double* cam = bal.camera(bal.obs_camera[i]);
        const double* pt = bal.point(bal.obs_point[i]);
        const double* obs = &bal.obs_pixel[2 * i];
        double r_ref[2], r_q[2];
        gba::snavely_residual(cam, pt, obs, r_ref);
        if (mode == "storage")
            residual_storage(cam, pt, obs, s_cam, s_pts, s_pix, r_q);
        else
            residual_arith(cam, pt, obs, s_cam, s_pts, s_pix, r_q);
        if (std::isnan(r_ref[0]) || std::isnan(r_ref[1])) { ++ref_nan; continue; }
        ref_cost2 += r_ref[0] * r_ref[0] + r_ref[1] * r_ref[1];
        double e = std::max(std::fabs(r_q[0] - r_ref[0]), std::fabs(r_q[1] - r_ref[1]));
        if (std::isnan(e)) e = 1e30;  // quantized path produced NaN; count it as garbage
        errs.push_back(e);
    }

    std::sort(errs.begin(), errs.end());
    auto pct = [&](double p) { return errs.empty() ? 0.0 : errs[std::min(errs.size() - 1, (size_t)(p * errs.size()))]; };
    long gt_half = 0, gt_5 = 0;
    for (double e : errs) {
        if (e > 0.5) ++gt_half;
        if (e > 5.0) ++gt_5;
    }

    std::printf(
        "{\n"
        "  \"file\": \"%s\",\n"
        "  \"precision\": \"%s\",\n"
        "  \"mode\": \"%s\",\n"
        "  \"scale\": \"%s\",\n"
        "  \"scales\": {\"cam\": %.6g, \"pts\": %.6g, \"pix\": %.6g},\n"
        "  \"observations\": %d,\n"
        "  \"reference_cost\": %.17g,\n"
        "  \"overflow_events\": %ld,\n"
        "  \"underflow_events\": %ld,\n"
        "  \"nan_events\": %ld,\n"
        "  \"residual_err_px\": {\"median\": %.6g, \"p90\": %.6g, \"p99\": %.6g, \"max\": %.6g},\n"
        "  \"residuals_err_gt_0.5px\": %ld,\n"
        "  \"residuals_err_gt_5px\": %ld\n"
        "}\n",
        file.c_str(), gba::quant::name(g_prec), mode.c_str(), scale.c_str(), s_cam, s_pts,
        s_pix, bal.num_observations, ref_cost2 / 2.0, g_stats.overflow, g_stats.underflow,
        g_stats.nan, pct(0.5), pct(0.9), pct(0.99),
        errs.empty() ? 0.0 : errs.back(), gt_half, gt_5);
    (void)ref_nan;
    return 0;
}
