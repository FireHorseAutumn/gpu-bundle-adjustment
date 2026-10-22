#pragma once

// Public interface to the GPU Levenberg-Marquardt solver (src/lm.cu).
// Parameters live in the caller's BALProblem (host, double) and are
// updated in place; updates stay FP64 end to end.

#include <string>

#include "gba/bal_problem.hpp"

namespace gba {

struct LMOptions {
    int max_iterations = 50;
    double initial_lambda = 1e-3;   // Marquardt diagonal damping
    double rel_decrease_tol = 1e-6; // convergence: relative cost decrease
    int max_inner_trials = 16;      // lambda escalations per outer iteration
    bool verbose = false;           // per-iteration log to stderr
};

struct LMResult {
    double initial_cost = 0.0;
    double final_cost = 0.0;
    int iterations = 0;             // outer iterations (incl. rejected steps)
    std::string termination;        // "CONVERGENCE" | "MAX_ITERATIONS" | "FAILURE"
    unsigned long long anomalies = 0;  // device-side NaN/Inf counter
    double total_time_s = 0.0;      // wall time of the solve itself
};

// Runs LM with GPU residual/Jacobian accumulation and a dense host-side
// reduced camera system (9C x 9C, Eigen LLT). Modifies prob's cameras and
// points to the solution. Throws std::runtime_error on CUDA failure.
LMResult solve_bal_gpu(BALProblem& prob, const LMOptions& opts = {});

// Host entry point to the same residual + analytic Jacobian the edge
// kernel uses (shared __host__ __device__ code). Jc is 2x9 row-major
// (camera: rodrigues[3], t[3], f, k1, k2), Jp is 2x3 row-major.
// Exposed for finite-difference validation in tests.
void edge_residual_jacobian_host(const double* camera9, const double* point3,
                                 const double* observed2, double* out_residual2,
                                 double* out_Jc, double* out_Jp);

}  // namespace gba
