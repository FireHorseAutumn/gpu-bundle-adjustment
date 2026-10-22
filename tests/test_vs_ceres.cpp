// test_vs_ceres — the GPU LM solver against the Ceres SPARSE_SCHUR oracle.
//
// Gates (per README TODO 6):
//   (a) model correctness: the solver's initial cost matches the reader's
//       total_squared_error / 2 to 1e-9 relative;
//   (b) same basin: final cost within 1e-3 relative of the Ceres oracle
//       (LM trajectories may differ);
//   (c) speed of convergence: <= 2x the oracle iteration count;
//   (d) no silent garbage: the device anomaly counter is zero.
// Plus a finite-difference check of the analytic edge Jacobian (the main
// theta branch; the theta <= 1e-12 first-order branch is not reachable by
// central differences at eps=1e-6 scale and is covered by the cost gates).

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>

#include "gba/bal_problem.hpp"
#include "gba/lm.hpp"

namespace {

int failures = 0;

void check(bool ok, const std::string& what) {
    std::fprintf(stderr, "%s: %s\n", ok ? "  ok" : "FAIL", what.c_str());
    if (!ok) ++failures;
}

void test_jacobian_numeric() {
    std::fprintf(stderr, "analytic vs finite-difference Jacobian\n");
    // (rodrigues, t, f, k1, k2), a point, a pixel — typical BAL magnitudes.
    const double cases[2][15] = {
        {0.1, -0.5, 0.3, 0.02, 0.01, -1.0, 500.0, -0.2, 0.01, 1.0, 2.0, 30.0, 100.0, -80.0,
         0.0},
        {1e-8, 2e-8, -1e-8, -0.05, 0.2, 0.4, 350.0, 0.1, -0.001, -2.0, 1.0, 25.0, -250.0,
         300.0, 0.0},
    };
    for (const auto& cs : cases) {
        double cam[9], pt[3], obs[2];
        for (int i = 0; i < 9; ++i) cam[i] = cs[i];
        for (int i = 0; i < 3; ++i) pt[i] = cs[9 + i];
        obs[0] = cs[12];
        obs[1] = cs[13];

        double r0[2], Jc_ref[18], Jp_ref[6];
        gba::edge_residual_jacobian_host(cam, pt, obs, r0, Jc_ref, Jp_ref);

        const double eps = 1e-6;
        double max_err = 0.0;
        auto fd = [&](double* params, int n, const double* Jref, int stride) {
            double junk_c[18], junk_p[6];
            for (int j = 0; j < n; ++j) {
                double saved = params[j];
                double rp[2], rm[2];
                params[j] = saved + eps;
                gba::edge_residual_jacobian_host(cam, pt, obs, rp, junk_c, junk_p);
                params[j] = saved - eps;
                gba::edge_residual_jacobian_host(cam, pt, obs, rm, junk_c, junk_p);
                params[j] = saved;
                for (int row = 0; row < 2; ++row) {
                    double num = (rp[row] - rm[row]) / (2.0 * eps);
                    double ana = Jref[row * stride + j];
                    double scale = std::max(1.0, std::abs(ana));
                    max_err = std::max(max_err, std::abs(num - ana) / scale);
                }
            }
        };
        fd(cam, 9, Jc_ref, 9);
        fd(pt, 3, Jp_ref, 3);
        check(max_err < 1e-5, "Jacobian rel err " + std::to_string(max_err) + " < 1e-5");
    }
}

void run_gate(const std::string& path, double oracle_cost, int oracle_iters,
              double reader_initial_sse) {
    std::fprintf(stderr, "== %s ==\n", path.c_str());
    gba::BALProblem prob = gba::load_bal(path);

    gba::LMOptions opts;
    gba::LMResult res = gba::solve_bal_gpu(prob, opts);

    std::fprintf(stderr,
                 "  initial %.9g final %.9g (oracle %.9g) iters %d (oracle %d) "
                 "termination %s anomalies %llu time %.3fs\n",
                 res.initial_cost, res.final_cost, oracle_cost, res.iterations,
                 oracle_iters, res.termination.c_str(), res.anomalies, res.total_time_s);

    double reader_initial_cost = reader_initial_sse / 2.0;
    check(std::fabs(res.initial_cost - reader_initial_cost) <=
              1e-9 * std::max(1.0, std::fabs(reader_initial_cost)),
          "(a) initial cost matches reader to 1e-9");
    check(std::fabs(res.final_cost - oracle_cost) <= 1e-3 * std::fabs(oracle_cost),
          "(b) final cost within 1e-3 of oracle");
    check(res.iterations <= 2 * oracle_iters, "(c) iterations <= 2x oracle");
    check(res.anomalies == 0, "(d) zero device anomalies");
    check(res.termination == "CONVERGENCE", "termination CONVERGENCE");
}

}  // namespace

int main() {
    test_jacobian_numeric();

    gba::BALProblem p16 = gba::load_bal("data/problem-16-22106-pre.txt.bz2");
    double sse16 = gba::total_squared_error(p16);
    gba::BALProblem lb = gba::load_bal("data/ladybug/problem-49-7776-pre.txt.bz2");
    double sse_lb = gba::total_squared_error(lb);

    run_gate("data/problem-16-22106-pre.txt.bz2", 18033.90351073163, 7, sse16);
    run_gate("data/ladybug/problem-49-7776-pre.txt.bz2", 13344.318399544838, 32, sse_lb);

    if (failures > 0) {
        std::fprintf(stderr, "%d gate(s) FAILED\n", failures);
        return 1;
    }
    std::printf("PASS\n");
    return 0;
}
