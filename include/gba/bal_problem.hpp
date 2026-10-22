#pragma once

#include <string>
#include <vector>

namespace gba {

// BAL problem in structure-of-arrays layout (Agarwal et al., ECCV 2010,
// https://grail.cs.washington.edu/projects/bal/).
//
// Camera parameter block (9 doubles, per camera):
//   [0..2] rotation as Rodrigues vector, [3..5] translation,
//   [6] focal length, [7] k1, [8] k2.
// Point block: 3 doubles. Observations reference cameras/points by index.
struct BALProblem {
    int num_cameras = 0;
    int num_points = 0;
    int num_observations = 0;

    std::vector<double> cameras;  // 9 * num_cameras
    std::vector<double> points;   // 3 * num_points

    std::vector<int> obs_camera;    // num_observations
    std::vector<int> obs_point;     // num_observations
    std::vector<double> obs_pixel;  // 2 * num_observations (x, y interleaved)

    const double* camera(int i) const { return cameras.data() + 9 * i; }
    double* camera(int i) { return cameras.data() + 9 * i; }
    const double* point(int i) const { return points.data() + 3 * i; }
    double* point(int i) { return points.data() + 3 * i; }
};

// Loads a BAL problem from a plain-text .txt file or a .bz2 file
// (streamed through bzcat). Throws std::runtime_error on any format
// violation or read failure.
BALProblem load_bal(const std::string& path);

// Reprojection residual for one observation, following the BAL camera
// model exactly (including the negation in p = -P / P.z):
//   P  = R(rodrigues) * X + t
//   p  = -P / P.z
//   r  = 1 + k1*|p|^2 + k2*|p|^4
//   px = f * r * p
// residual = px - observed_pixel. Writes 2 doubles into out.
void snavely_residual(const double* camera9, const double* point3,
                      const double* observed2, double* out2);

// Sum of squared residuals over all observations (the unrobustified
// BAL objective, i.e. what Ceres reports as cost * 2).
double total_squared_error(const BALProblem& p);

}  // namespace gba
