// test_launch_bounds — regression for the point-count launch boundary in
// the LM solver. grid_for() caps launches at 1024 blocks (262144 threads);
// the edge/cost kernels grid-stride past the cap and damp_invert_v_kernel
// must too. Synthetic noise-free problem with P > 262144: if any point's
// damped V inverse is never written, that point is frozen (or its garbage
// inverse corrupts the reduced system) and the cost cannot reach ~0.

#include <cmath>
#include <cstdio>
#include <random>
#include <string>

#include "gba/bal_problem.hpp"
#include "gba/lm.hpp"

namespace {

int failures = 0;

void check(bool ok, const std::string& what) {
    std::fprintf(stderr, "%s: %s\n", ok ? "  ok" : "FAIL", what.c_str());
    if (!ok) ++failures;
}

}  // namespace

int main() {
    constexpr int C = 8;
    constexpr int P = 400000;  // > 1024 * 256, the grid_for launch cap
    std::mt19937_64 rng(7);
    std::uniform_real_distribution<double> xy(-3.0, 3.0), zz(4.0, 12.0),
        pert(-0.05, 0.05);

    gba::BALProblem truth, prob;
    truth.num_cameras = prob.num_cameras = C;
    truth.num_points = prob.num_points = P;
    truth.num_observations = prob.num_observations = 2 * P;

    truth.cameras.resize(9 * C);
    for (int i = 0; i < C; ++i) {
        double* c = truth.camera(i);
        c[0] = 0.02 * (i + 1);
        c[1] = -0.01 * i;
        c[2] = 0.03;
        c[3] = 0.3 * i;
        c[4] = -0.1 * i;
        c[5] = 0.05 * i;
        c[6] = 500.0 + 10.0 * i;
        c[7] = -0.05;
        c[8] = 0.001;
    }
    truth.points.resize(3 * P);
    for (int k = 0; k < P; ++k) {
        truth.points[3 * k] = xy(rng);
        truth.points[3 * k + 1] = xy(rng);
        truth.points[3 * k + 2] = zz(rng);
    }

    // Noise-free observations from the truth; the solve starts from
    // perturbed points, so cost 0 is achievable only if every point moves.
    prob.cameras = truth.cameras;
    prob.points = truth.points;
    for (int k = 0; k < P; ++k)
        for (int j = 0; j < 3; ++j) prob.points[3 * k + j] += pert(rng);
    prob.obs_camera.resize(2 * P);
    prob.obs_point.resize(2 * P);
    prob.obs_pixel.resize(4 * P);
    for (int k = 0; k < P; ++k) {
        const int cams[2] = {k % C, (k + 1) % C};
        for (int m = 0; m < 2; ++m) {
            const int e = 2 * k + m;
            prob.obs_camera[e] = cams[m];
            prob.obs_point[e] = k;
            double pred[2];
            gba::snavely_residual(truth.camera(cams[m]), truth.point(k),
                                  prob.obs_pixel.data() + 2 * e, pred);
            prob.obs_pixel[2 * e] = pred[0];
            prob.obs_pixel[2 * e + 1] = pred[1];
        }
    }

    gba::LMResult res = gba::solve_bal_gpu(prob);
    std::fprintf(stderr,
                 "P=%d E=%d: initial %.9g final %.9g iters %d termination %s "
                 "anomalies %llu\n",
                 P, 2 * P, res.initial_cost, res.final_cost, res.iterations,
                 res.termination.c_str(), res.anomalies);

    check(res.initial_cost > 1.0, "perturbation live (initial cost > 1)");
    check(res.termination == "CONVERGENCE", "termination CONVERGENCE");
    check(res.anomalies == 0, "zero device anomalies");
    // Pre-fix plateau sits at the frozen points' share of the cost
    // (~34% here); a correct solve is noise-free-quadratic to ~0.
    check(res.final_cost < 1e-4 * res.initial_cost,
          "final cost < 1e-4 of initial (no frozen points)");

    if (failures > 0) {
        std::fprintf(stderr, "%d gate(s) FAILED\n", failures);
        return 1;
    }
    std::printf("PASS\n");
    return 0;
}
