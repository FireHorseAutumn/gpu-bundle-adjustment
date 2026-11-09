# Record

The full record for gpu-bundle-adjustment (A3 BAL-scale GPU BA): the claims, the kill criterion,
the precision experiment, both BAL tables with every caveat, and the fix
path. The README carries the headline; this file carries the evidence.
Small-problem table: interleaved best-of-7, Ryzen 9 8940HX (32 threads,
powersave governor; absolute numbers move ~2x with CPU frequency state) +
RTX 5060 Laptop (sm_120), CUDA 13.2, FP64 throughout. Scaling table: seven
interleaved pairs per dataset, idle machine, governor and platform profile
pinned to performance, benchmarks confined to 15 physical cores (Ceres
threads=30, last core quarantined for a busy-polling input daemon), kernel
7.2.15.

## Claims

- Edge-parallel Schur complement beats Ceres SPARSE_SCHUR on BAL-scale and
  streaming/windowed problems.
- Fused residual/J^T J accumulation wins by memory traffic alone.

## Kill criterion

- If Ceres wins on BAL datasets at any size you test, publish the table. The
  old pipeline's lesson: Ceres did 40k residuals in 0.26 s: algorithmic
  improvement beats hardware. Do not benchmark against a 4-camera toy.

## Precision experiment: FP8/FP4 storage, DEAD, negative result

Question: can problem data live in FP8 (e4m3/e5m2) or FP4 (e2m1) with
per-tensor scale factors, TRT-style, while the optimizer runs in FP32/FP64?
Tested at storage level only (quantize BAL inputs, run unmodified Ceres
oracle); no GPU kernel was built.

Method: `tools/quantize_bal.cu` (hardware-exact grids via CUDA 13.2
`cuda_fp8.h`/`cuda_fp4.h` host conversions) + `tools/fp8_residual_probe.cu`
(per-observation residual error vs the double reference, explicit
overflow/NaN counting). Raw results: `data/quantized/results_*.jsonl`.

Verdict: **gate failed by 2 to 3 orders of magnitude; do not build the GPU
variant.**

| dataset (oracle cost) | precision | scale | final cost | result |
|---|---|---|---|---|
| p16 (18033.9) | fp32 | - | 18033.90 | control ✓ |
| p16 | fp16 | none | 18298.02 | SURVIVES (+1.5%, same 7 iters) |
| p16 | bf16 | none | 36441.58 | converged, 2× cost (borderline) |
| p16 | e4m3 | auto | 4,425,457 | converged to noise floor (245×) |
| p16 | e5m2 | auto | 16,622,552 | converged to noise floor (922×) |
| p16 | fp4 | any | - | NO_CONVERGENCE |
| ladybug-49 (13344.3) | fp32 | - | 13344.32 | control ✓ |
| ladybug-49 | fp16 | none | 13368.57 | SURVIVES (+0.2%, same 32 iters) |
| ladybug-49 | bf16 | none | 14960.72 | converged, +12%, 43 iters |
| ladybug-49 | e4m3 | auto | 367,630 | NO_CONVERGENCE (27×) |
| ladybug-49 | e5m2 | auto | 1,376,951 | NO_CONVERGENCE (103×) |
| ladybug-49 | fp4 | any | - | FAILURE (Ceres aborts) |

Why it cannot work: BAL initial residuals are already ~7 px rms. e4m3 has a
12.5% relative ulp, so at |pixel| ≈ 350 the quantization step is ~32 px:
probe-measured median residual corruption 23.8 px (3.4× the signal). The
optimizer fits the quantization noise; the cost floor *is* the noise.
Scaling fixes range only; relative error is scale-invariant, so per-tensor
and per-channel (`auto9`) scales both fail identically. e5m2 (2 mantissa
bits) is strictly worse than e4m3 despite the bigger range; fp4 e2m1
(1 mantissa bit: 2 grid values per octave, 7 positive values total,
max 6.0) is coarser than the data itself.

The boundary is mantissa bits vs the noise floor: fp16 (10 bits -> ~0.2 px
error at |pixel| ≈ 400) sits under the 7 px signal and survives; bf16
(8 bits -> ~1.5 px) grazes it and pays 1.1 to 2× cost; e4m3 (3 bits ->
~24 px) exceeds it and collapses. The floor is absolute, set by pixel
magnitudes: a from-scratch problem with larger initial residuals would
stall at the same quantization floor after early progress.

Consequence: FP8/FP4 storage is dead for BA, but **fp16 storage of
problem data with fp32+ compute is viable**: halves memory traffic on the
observation arrays (the dominant footprint) at ≤1.5% cost. Available as a
kernel-level option when TODO 4 to 6 land, if bandwidth ever binds.

The lesson rhymes with the archived TensorRT detector failure: reduced
precision in measurement space silently relocates the objective's minimum.
The GPU solver (TODO 4 to 6) proceeds in FP32/FP64.

## GPU LM solver: vertical slice landed, beats Ceres on both test datasets

The solver TODOs are done: `include/gba/schur.cuh` (batched 9x9 Cholesky
factor/solve + 3x3 SPD inverse, one thread per block, non-SPD fails
loudly), `src/lm.cu` (LM with per-observation analytic-Jacobian edge
kernel; atomic accumulation into U/V/W blocks; Marquardt diagonal damping
with Nielsen gain-ratio accept/reject; device anomaly counter for
non-finite residuals/Jacobians and non-SPD V blocks), `tests/` (batched
Cholesky vs Eigen; finite-difference Jacobian check; cost/iteration gates
vs the Ceres oracle; launch-boundary regression at P > 262144),
`benchmarks/bench_bal.cpp` (same JSON schema as bench_ceres).

v1 scope decision: the reduced camera system S is dense (9C x 9C) and
solved on the host with Eigen LLT: fine at C=16/49. S assembly runs on
the host with per-thread privatized S copies (shared-S accumulation
measured ~5x slower from cache-line ping-pong on the 9x9 block RMWs).
Follow-ups, in order: sparse S (block CSR over the camera graph), S
assembly on device, cuSOLVER potrf for the factorization. The dense-S
path was estimated to break around C~400; measured, it already loses at
C=257 (scaling table below).

### BAL table: honest win/loss (TODO 8)

| dataset | solver | final cost | iters | total s | per-iter s |
|---|---|---|---|---|---|
| problem-16 (16 cam / 22106 pt / 83718 obs) | Ceres SPARSE_SCHUR | 18033.903510731627 | 7 | 1.509 | 0.216 |
| problem-16 | gba-gpu-lm | 18033.903480853227 | 9 | 0.336 | 0.037 |
| ladybug problem-49 (49 cam / 7776 pt / 31843 obs) | Ceres SPARSE_SCHUR | 13344.318399544838 | 32 | 1.616* | 0.050 |
| ladybug problem-49 | gba-gpu-lm | 13344.269750634063 | 22 | 0.780 | 0.035 |

\* ladybug Ceres best across 12 interleaved runs (2.246 median over the
first 5; the 7-run round never beat 1.776: the 1.616 came from the
earlier 5-run round, same conditions).

GPU wins both. Caveats, honestly:

- The earlier-recorded Ceres oracle runs (committed JSONs) were faster:
  0.552 s (p16) and 1.357 s (ladybug), than anything reproducible in
  that session (CPU frequency state). The GPU best still beats those
  numbers on both datasets.
- The win is the edge kernel, not the Schur solve: GPU residual+analytic
  Jacobian+J^T J accumulation costs ~0.06 s per whole solve (vs Ceres's
  autodiff residual evaluation dominating its timeline); host-side dense
  S assembly + LLT is ~60 to 80% of the GPU solver's time. At these sizes
  the total is small enough that the CPU-side Schur still wins the
  overall race.
- Medians are much closer than bests (p16: 0.91 vs 2.23; ladybug: 1.66
  vs 2.24): the GPU solver is more consistent because most of its work
  is off the frequency-lottery CPU.
- Phase breakdown is available via `bench_bal --verbose` (edge-accumulate
  / damp+Vinv / S assembly / LLT+backsub / trial cost).

## Scaling table: the win does not survive scale

Seven interleaved Ceres/GPU pairs per dataset, idle machine, governor and
platform profile pinned to performance. Benchmarks confined to 15 physical
cores (Ceres threads=30); the last core is quarantined for a busy-polling
input daemon. Kernel 7.2.15, CUDA 13.2, FP64 throughout. Round 1 of each
dataset caught post-boot OS housekeeping: kept in the range, disclosed,
not deleted. Raw JSONs in `build/scale-runs/` (regenerable), summary in
`data/scale_2026-11-07_summary.json`.

| dataset | solver | final cost | iters | best s | median s | worst s |
|---|---|---|---|---|---|---|
| trafalgar problem-257 (257 cam / 65132 pt / 225911 obs) | Ceres SPARSE_SCHUR | 96656.0529 | 51 | 17.168 | 17.654 | 39.940* |
| trafalgar problem-257 | gba-gpu-lm | 96519.6475 | 50 | 30.143 | 30.646 | 196.165* |
| dubrovnik problem-356 (356 cam / 226730 pt / 1255268 obs) | Ceres SPARSE_SCHUR | 493930.9658 | 13 | 24.813 | 25.782 | 27.660 |
| dubrovnik problem-356 | gba-gpu-lm | 490690.4308 | 22 | 70.564 | 71.735 | 238.302* |

\* round 1 of 7, post-boot housekeeping. Rounds 2 to 7 span 17.17 to 18.53
(Ceres 257), 30.14 to 31.67 (GPU 257), 24.81 to 27.66 (Ceres 356),
70.56 to 73.56 (GPU 356).

The GPU loses at scale: 1.74x slower than Ceres on Trafalgar, 2.78x on
Dubrovnik (medians). The mechanism is the one the v1 scope note predicted:
dense host-side S is 9C x 9C, and assembly plus LLT already dominate at
C=257. The edge kernel is not the bottleneck. The CPU is.

Two facts that keep this honest:

- Every Trafalgar Ceres run hit the iteration cap (51 steps,
  NO_CONVERGENCE), cost flat at 96656.0529, on a quiet machine. That is
  Ceres solver behavior on this problem, not a measurement artifact.
- The GPU solver reaches the lower final cost on both datasets
  (96519.6 vs 96656.1; 490690.4 vs 493931.0). Its LM trajectory is
  better; its host linear algebra is slower. The fix path is unchanged:
  sparse block-CSR S, device assembly, cuSOLVER potrf.

An earlier scaling attempt ran under machine contention. None of its
numbers are published. The 16/49-camera table above stands as measured,
with its powersave caveat.

## Layout and build notes

- `include/gba/`: problem definition, edge/camera types (headers)
- `src/`: Schur kernels (.cu), LM driver, BAL reader
- `tests/`: correctness vs Ceres on small BAL subsets
- `benchmarks/`: BAL leaderboard-style table; JSON out
- `data/`: BAL download script (public datasets)

No sibling repos; everything comes from public sources: CUDA toolkit,
Eigen, Ceres, BAL. The batched 9x9 Cholesky for the camera blocks lives
in-repo (`schur.cuh`). If the default g++ is newer than nvcc supports,
CMake prefers `g++-15` as the CUDA host compiler, falling back to
`-allow-unsupported-compiler` with `-std=c++20`. If no BLAS dev package
is installed, CMake points Ceres's FindBLAS at the OpenBLAS runtime
library directly.

## TODO (in order)

1. [x] `data/fetch_bal.sh`: pull BAL datasets (`smoke` | dataset names | `all` | URL)
2. [x] `src/bal_reader.cpp`: load BAL (.txt or .bz2) into SoA (`include/gba/bal_problem.hpp`)
3. [x] `benchmarks/bench_ceres.cpp`: Ceres SPARSE_SCHUR baseline + oracle:
      cross-checks the reader's initial cost against Ceres (exit 2 on mismatch),
      JSON out. Baseline uses all cores by default (`--threads` to override).
4. [x] `include/gba/schur.cuh`: block-diagonal C inverse; 9x9 camera blocks,
      batched Cholesky written in-repo
5. [x] `src/lm.cu`: LM loop with GPU Schur
6. [x] `tests/test_vs_ceres.cpp`: match Ceres cost on BAL subset
      (bench_ceres already provides the oracle numbers)
7. [x] `benchmarks/bench_bal.cpp`: full BAL table; JSON out
8. [x] README: BAL table with honest win/loss per dataset
