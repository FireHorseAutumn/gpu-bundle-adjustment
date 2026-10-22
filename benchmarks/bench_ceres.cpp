// bench_ceres — Ceres SPARSE_SCHUR baseline on BAL problems.
//
// This is the oracle and the baseline column for the whole project:
// every GPU kernel that lands must reproduce these costs, and every
// benchmark table compares against these times.
//
// Usage:
//   bench_ceres <problem.txt[.bz2]> [options]
//     --iterations=N   max LM iterations        (default 50)
//     --loss=none|huber                         (default none)
//     --delta=D        Huber delta              (default 1.0)
//     --threads=N      Ceres threads            (default 0 = hardware)
//     --json=PATH      also write results as JSON
//     --verbose        print Ceres full report
//
// Exit code 2 if our own reader's initial cost disagrees with Ceres —
// that means the reader or camera model is wrong, not the optimizer.

#include <ceres/ceres.h>
#include <ceres/rotation.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>

#include "gba/bal_problem.hpp"

namespace {

// Canonical Snavely reprojection error, identical to the one in the
// Ceres examples and the BAL paper.
struct SnavelyReprojectionError {
    SnavelyReprojectionError(double x, double y) : observed_x(x), observed_y(y) {}

    template <typename T>
    bool operator()(const T* const camera, const T* const point, T* residuals) const {
        T p[3];
        ceres::AngleAxisRotatePoint(camera, point, p);
        p[0] += camera[3];
        p[1] += camera[4];
        p[2] += camera[5];

        T xp = -p[0] / p[2];
        T yp = -p[1] / p[2];

        const T& k1 = camera[7];
        const T& k2 = camera[8];
        T r2 = xp * xp + yp * yp;
        T distortion = T(1.0) + r2 * (k1 + k2 * r2);

        T predicted_x = camera[6] * distortion * xp;
        T predicted_y = camera[6] * distortion * yp;

        residuals[0] = predicted_x - T(observed_x);
        residuals[1] = predicted_y - T(observed_y);
        return true;
    }

    double observed_x, observed_y;
};

struct Args {
    std::string file;
    int iterations = 50;
    std::string loss = "none";
    double delta = 1.0;
    int threads = 0;
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
        else if (std::string v = eat("--loss="); !v.empty())
            a.loss = v;
        else if (std::string v = eat("--delta="); !v.empty())
            a.delta = std::stod(v);
        else if (std::string v = eat("--threads="); !v.empty())
            a.threads = std::stoi(v);
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
        std::fprintf(stderr, "usage: bench_ceres <problem.txt[.bz2]> [options]\n");
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

    // Cross-check: our reader + camera model must reproduce Ceres' cost
    // at the initial parameters. Ceres reports cost = sum rho(SSE_i) / 2.
    const double own_sse = gba::total_squared_error(bal);
    double own_cost = own_sse / 2.0;
    if (args.loss == "huber") {
        // Huber on the squared norm, as Ceres applies it: with s = |r|^2,
        // rho(s) = s for s <= d^2, else 2*d*sqrt(s) - d^2.
        own_cost = 0.0;
        double res[2];
        for (int i = 0; i < bal.num_observations; ++i) {
            gba::snavely_residual(bal.camera(bal.obs_camera[i]), bal.point(bal.obs_point[i]),
                                  &bal.obs_pixel[2 * i], res);
            double s = res[0] * res[0] + res[1] * res[1];
            double d2 = args.delta * args.delta;
            own_cost += 0.5 * (s <= d2 ? s : 2.0 * args.delta * std::sqrt(s) - d2);
        }
    }

    // Heap-allocated: Ceres Problem takes ownership of the loss function.
    ceres::LossFunction* loss = nullptr;
    if (args.loss == "huber")
        loss = new ceres::HuberLoss(args.delta);
    else if (args.loss != "none") {
        std::fprintf(stderr, "unknown loss: %s\n", args.loss.c_str());
        return 1;
    }

    ceres::Problem problem;
    for (int i = 0; i < bal.num_observations; ++i) {
        problem.AddResidualBlock(
            new ceres::AutoDiffCostFunction<SnavelyReprojectionError, 2, 9, 3>(
                new SnavelyReprojectionError(bal.obs_pixel[2 * i], bal.obs_pixel[2 * i + 1])),
            loss, bal.camera(bal.obs_camera[i]), bal.point(bal.obs_point[i]));
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::SPARSE_SCHUR;
    options.max_num_iterations = args.iterations;
    options.minimizer_progress_to_stdout = args.verbose;
    if (args.threads > 0)
        options.num_threads = args.threads;
    else
        options.num_threads = static_cast<int>(std::thread::hardware_concurrency());

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    if (std::fabs(summary.initial_cost - own_cost) >
        1e-9 * std::max(1.0, std::fabs(own_cost))) {
        std::fprintf(stderr,
                     "MISMATCH: reader cost %.17g vs Ceres initial cost %.17g\n"
                     "reader or camera model is broken; benchmark results void\n",
                     own_cost, summary.initial_cost);
        return 2;
    }

    const double total_s = summary.total_time_in_seconds;
    const int iters = static_cast<int>(summary.iterations.size());
    const double per_iter_s = iters > 0 ? total_s / iters : 0.0;

    if (args.verbose) std::fprintf(stderr, "%s", summary.FullReport().c_str());

    char json[1024];
    std::snprintf(json, sizeof json,
                  "{\n"
                  "  \"file\": \"%s\",\n"
                  "  \"solver\": \"ceres SPARSE_SCHUR\",\n"
                  "  \"loss\": \"%s\",\n"
                  "  \"num_cameras\": %d,\n"
                  "  \"num_points\": %d,\n"
                  "  \"num_observations\": %d,\n"
                  "  \"initial_cost\": %.17g,\n"
                  "  \"final_cost\": %.17g,\n"
                  "  \"iterations\": %d,\n"
                  "  \"total_time_s\": %.6f,\n"
                  "  \"per_iteration_s\": %.6f,\n"
                  "  \"termination\": \"%s\",\n"
                  "  \"threads\": %d\n"
                  "}",
                  args.file.c_str(), args.loss.c_str(), bal.num_cameras, bal.num_points,
                  bal.num_observations, summary.initial_cost, summary.final_cost, iters,
                  total_s, per_iter_s, ceres::TerminationTypeToString(summary.termination_type),
                  summary.num_threads_used);
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
