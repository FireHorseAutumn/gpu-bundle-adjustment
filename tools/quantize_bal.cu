// quantize_bal — quantize a BAL problem's cameras/points/observations onto
// an FP32/FP8/FP4 grid and write a new BAL .txt, for running through the
// existing bench_ceres oracle. This tests STORAGE quantization: what the
// optimizer sees when the problem data lives in low precision.
//
// Usage:
//   quantize_bal <in.txt[.bz2]> <out.txt> --precision=fp32|e4m3|e5m2|fp4
//                [--scale=none|auto|auto9|VALUE]
//
// Scale modes (per quantize(v*s)/s):
//   none   — raw values onto the grid (default; expect e4m3/fp4 overflow)
//   auto   — one scale per array (cameras, points, pixels)
//   auto9  — cameras get one scale per parameter slot (9 scales, the
//            per-channel analogue); points/pixels per-array
//   VALUE  — fixed scale for all arrays
// Auto scales map the array's max |v| to 90% of the format's max.

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

struct Args {
    std::string in, out;
    Precision precision = Precision::E4M3;
    std::string scale = "none";
};

Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string s = argv[i];
        auto eat = [&](const char* key) -> std::string {
            return s.rfind(key, 0) == 0 ? s.substr(std::strlen(key)) : "";
        };
        if (std::string v = eat("--precision="); !v.empty()) {
            if (v == "fp32") a.precision = Precision::FP32;
            else if (v == "fp16") a.precision = Precision::FP16;
            else if (v == "bf16") a.precision = Precision::BF16;
            else if (v == "e4m3") a.precision = Precision::E4M3;
            else if (v == "e5m2") a.precision = Precision::E5M2;
            else if (v == "fp4") a.precision = Precision::FP4;
            else { std::fprintf(stderr, "bad precision: %s\n", v.c_str()); std::exit(1); }
        } else if (std::string v = eat("--scale="); !v.empty()) {
            a.scale = v;
        } else if (s.rfind("--", 0) != 0 && a.in.empty()) {
            a.in = s;
        } else if (s.rfind("--", 0) != 0 && a.out.empty()) {
            a.out = s;
        } else {
            std::fprintf(stderr, "bad argument: %s\n", s.c_str());
            std::exit(1);
        }
    }
    if (a.in.empty() || a.out.empty()) {
        std::fprintf(stderr,
                     "usage: quantize_bal <in> <out> --precision=P [--scale=none|auto|auto9|VALUE]\n");
        std::exit(1);
    }
    return a;
}

double maxabs(const std::vector<double>& v) {
    double m = 0.0;
    for (double x : v) m = std::max(m, std::fabs(x));
    return m;
}

// Quantize a vector with one scale. Returns stats; error tracked against
// original values.
Stats quantize_array(std::vector<double>& v, Precision p, double s,
                     double* max_rel_err, double* sum_rel_err, long* rel_count) {
    Stats st;
    *max_rel_err = 0.0;
    *sum_rel_err = 0.0;
    *rel_count = 0;
    for (double& x : v) {
        const double orig = x;
        x = gba::quant::roundtrip(p, orig * s, &st) / s;
        const double denom = std::max(1e-12, std::fabs(orig));
        const double rel = std::fabs(x - orig) / denom;
        *max_rel_err = std::max(*max_rel_err, rel);
        *sum_rel_err += rel;
        ++*rel_count;
    }
    return st;
}

void write_bal(const gba::BALProblem& p, const std::string& path) {
    FILE* fp = std::fopen(path.c_str(), "w");
    if (!fp) {
        std::fprintf(stderr, "cannot write %s\n", path.c_str());
        std::exit(1);
    }
    std::fprintf(fp, "%d %d %d\n", p.num_cameras, p.num_points, p.num_observations);
    for (int i = 0; i < p.num_observations; ++i)
        std::fprintf(fp, "%d %d %.17g %.17g\n", p.obs_camera[i], p.obs_point[i],
                     p.obs_pixel[2 * i], p.obs_pixel[2 * i + 1]);
    for (int i = 0; i < p.num_cameras; ++i) {
        for (int j = 0; j < 9; ++j) std::fprintf(fp, "%.17g%c", p.cameras[9 * i + j], j == 8 ? '\n' : ' ');
    }
    for (int i = 0; i < p.num_points; ++i)
        std::fprintf(fp, "%.17g %.17g %.17g\n", p.points[3 * i], p.points[3 * i + 1],
                     p.points[3 * i + 2]);
    std::fclose(fp);
}

}  // namespace

int main(int argc, char** argv) {
    Args args = parse_args(argc, argv);
    gba::BALProblem bal = gba::load_bal(args.in);

    const double maxr = gba::quant::max_representable(args.precision);
    const bool scalable = args.precision != Precision::FP32 && args.scale != "none";

    // Resolve scales.
    double s_cam = 1.0, s_pts = 1.0, s_pix = 1.0;
    std::vector<double> s_cam9;  // per-slot camera scales, auto9 only
    if (scalable) {
        if (args.scale == "auto" || args.scale == "auto9") {
            s_cam = 0.9 * maxr / std::max(1e-12, maxabs(bal.cameras));
            s_pts = 0.9 * maxr / std::max(1e-12, maxabs(bal.points));
            s_pix = 0.9 * maxr / std::max(1e-12, maxabs(bal.obs_pixel));
            if (args.scale == "auto9") {
                s_cam9.resize(9);
                for (int j = 0; j < 9; ++j) {
                    double m = 0.0;
                    for (int i = 0; i < bal.num_cameras; ++i)
                        m = std::max(m, std::fabs(bal.cameras[9 * i + j]));
                    s_cam9[j] = 0.9 * maxr / std::max(1e-12, m);
                }
            }
        } else {
            s_cam = s_pts = s_pix = std::stod(args.scale);
        }
    }

    struct ArrayReport {
        const char* name;
        Stats st;
        double max_rel, mean_rel, maxabs_before, scale;
    };
    std::vector<ArrayReport> reports;

    auto run_array = [&](const char* name, std::vector<double>& v, double s) {
        double mrel, srel; long n;
        Stats st = quantize_array(v, args.precision, s, &mrel, &srel, &n);
        reports.push_back({name, st, mrel, srel / n, 0.0, s});
    };

    if (!s_cam9.empty()) {
        // Per-slot: quantize each of the 9 camera parameter lanes separately.
        Stats agg; double mrel = 0.0, srel = 0.0; long n = 0;
        for (int j = 0; j < 9; ++j) {
            for (int i = 0; i < bal.num_cameras; ++i) {
                double& x = bal.cameras[9 * i + j];
                const double orig = x;
                x = gba::quant::roundtrip(args.precision, orig * s_cam9[j], &agg) / s_cam9[j];
                const double rel = std::fabs(x - orig) / std::max(1e-12, std::fabs(orig));
                mrel = std::max(mrel, rel); srel += rel; ++n;
            }
        }
        reports.push_back({"cameras(per-slot)", agg, mrel, srel / n, 0.0, -1.0});
    } else {
        run_array("cameras", bal.cameras, s_cam);
    }
    run_array("points", bal.points, s_pts);
    run_array("pixels", bal.obs_pixel, s_pix);

    // maxabs before quantization for context (recompute from the reports' scale where auto).
    write_bal(bal, args.out);

    std::printf("{\n  \"file_in\": \"%s\",\n  \"file_out\": \"%s\",\n  \"precision\": \"%s\",\n  \"scale_mode\": \"%s\",\n",
                args.in.c_str(), args.out.c_str(), gba::quant::name(args.precision), args.scale.c_str());
    for (size_t i = 0; i < reports.size(); ++i) {
        const auto& r = reports[i];
        std::printf(
            "  \"%s\": {\"scale\": %.6g, \"overflow\": %ld, \"underflow\": %ld, \"nan\": %ld, "
            "\"max_rel_err\": %.6g, \"mean_rel_err\": %.6g}%s\n",
            r.name, r.scale, r.st.overflow, r.st.underflow, r.st.nan, r.max_rel, r.mean_rel,
            i + 1 < reports.size() ? "," : "");
    }
    std::printf("}\n");
    return 0;
}
