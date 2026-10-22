// bench_bal — GPU LM (gba-gpu-lm) on BAL problems; the GPU column for the
// benchmark table, against bench_ceres as oracle/baseline.
//
// Usage:
//   bench_bal <problem.txt[.bz2]> [options]
//     --iterations=N   max LM iterations        (default 50)
//     --lambda=D       initial damping          (default 1e-3)
//     --json=PATH      also write results as JSON
//     --verbose        print per-iteration log
//
// JSON schema matches bench_ceres (minus "threads", plus "device").

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>

#include <cuda_runtime.h>

#include "gba/bal_problem.hpp"
#include "gba/lm.hpp"

namespace {

struct Args {
    std::string file;
    int iterations = 50;
    double lambda = 1e-3;
    std::string json;
    bool verbose = false;
};

Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string s = argv[i];
        auto eat = [&](const char* key) -> std::string {
            return s.rfind(key, 0) == 0 ? s.substr(std::strlen(key)) : "";
        };
        if (s == "--verbose")
            a.verbose = true;
        else if (std::string v = eat("--iterations="); !v.empty())
            a.iterations = std::stoi(v);
        else if (std::string v = eat("--lambda="); !v.empty())
            a.lambda = std::stod(v);
        else if (std::string v = eat("--json="); !v.empty())
            a.json = v;
        else if (s.rfind("--", 0) != 0 && a.file.empty())
            a.file = s;
        else {
            std::fprintf(stderr, "bad argument: %s\n", s.c_str());
            std::exit(1);
        }
    }
    if (a.file.empty()) {
        std::fprintf(stderr, "usage: bench_bal <problem.txt[.bz2]> [options]\n");
        std::exit(1);
    }
    return a;
}

}  // namespace

int main(int argc, char** argv) {
    Args args = parse_args(argc, argv);

    gba::BALProblem bal = gba::load_bal(args.file);
    std::fprintf(stderr, "loaded %s: %d cameras, %d points, %d observations\n",
                 args.file.c_str(), bal.num_cameras, bal.num_points, bal.num_observations);

    // Model cross-check: the solver's initial cost must reproduce the
    // reader's (exit 2 on mismatch, same contract as bench_ceres).
    const double own_cost = gba::total_squared_error(bal) / 2.0;

    cudaDeviceProp prop{};
    int dev = 0;
    cudaGetDevice(&dev);
    cudaGetDeviceProperties(&prop, dev);

    gba::LMOptions opts;
    opts.max_iterations = args.iterations;
    opts.initial_lambda = args.lambda;
    opts.verbose = args.verbose;
    gba::LMResult res = gba::solve_bal_gpu(bal, opts);

    if (std::fabs(res.initial_cost - own_cost) >
        1e-9 * std::max(1.0, std::fabs(own_cost))) {
        std::fprintf(stderr,
                     "MISMATCH: reader cost %.17g vs solver initial cost %.17g\n"
                     "camera model is broken; benchmark results void\n",
                     own_cost, res.initial_cost);
        return 2;
    }
    if (res.anomalies != 0) {
        std::fprintf(stderr, "device anomaly counter is %llu — refusing to report\n",
                     res.anomalies);
        return 3;
    }

    const double per_iter_s =
        res.iterations > 0 ? res.total_time_s / res.iterations : 0.0;

    char json[1024];
    std::snprintf(json, sizeof json,
                  "{\n"
                  "  \"file\": \"%s\",\n"
                  "  \"solver\": \"gba-gpu-lm\",\n"
                  "  \"loss\": \"none\",\n"
                  "  \"num_cameras\": %d,\n"
                  "  \"num_points\": %d,\n"
                  "  \"num_observations\": %d,\n"
                  "  \"initial_cost\": %.17g,\n"
                  "  \"final_cost\": %.17g,\n"
                  "  \"iterations\": %d,\n"
                  "  \"total_time_s\": %.6f,\n"
                  "  \"per_iteration_s\": %.6f,\n"
                  "  \"termination\": \"%s\",\n"
                  "  \"device\": \"%s\"\n"
                  "}",
                  args.file.c_str(), bal.num_cameras, bal.num_points, bal.num_observations,
                  res.initial_cost, res.final_cost, res.iterations, res.total_time_s,
                  per_iter_s, res.termination.c_str(), prop.name);
    std::printf("%s\n", json);

    if (!args.json.empty()) {
        FILE* fp = std::fopen(args.json.c_str(), "w");
        if (!fp) {
            std::fprintf(stderr, "cannot write %s\n", args.json.c_str());
            return 1;
        }
        std::fprintf(fp, "%s\n", json);
        std::fclose(fp);
    }
    return 0;
}
