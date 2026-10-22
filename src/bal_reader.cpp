#include "gba/bal_problem.hpp"

#include <charconv>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>

namespace gba {
namespace {

std::string read_all(const std::string& path) {
    std::string cmd;
    FILE* fp = nullptr;
    if (path.size() >= 4 && path.compare(path.size() - 4, 4, ".bz2") == 0) {
        cmd = "bzcat -- \"" + path + "\"";
        fp = popen(cmd.c_str(), "r");
    } else {
        fp = std::fopen(path.c_str(), "rb");
    }
    if (!fp) throw std::runtime_error("cannot open " + path);

    std::string buf;
    char chunk[1 << 20];
    size_t n;
    while ((n = std::fread(chunk, 1, sizeof chunk, fp)) > 0)
        buf.append(chunk, n);

    int rc = (path.size() >= 4 && path.compare(path.size() - 4, 4, ".bz2") == 0)
                 ? pclose(fp)
                 : std::fclose(fp);
    if (rc != 0) throw std::runtime_error("read failed for " + path);
    return buf;
}

class Parser {
public:
    explicit Parser(const std::string& text) : cur_(text.data()), end_(cur_ + text.size()) {}

    template <typename T>
    T next() {
        skip_ws();
        T v{};
        auto [ptr, ec] = std::from_chars(cur_, end_, v);
        if (ec != std::errc()) throw std::runtime_error("malformed BAL number");
        cur_ = ptr;
        return v;
    }

    bool at_end() {
        skip_ws();
        return cur_ == end_;
    }

private:
    void skip_ws() {
        while (cur_ != end_ && (*cur_ == ' ' || *cur_ == '\n' || *cur_ == '\t' || *cur_ == '\r'))
            ++cur_;
    }
    const char* cur_;
    const char* end_;
};

}  // namespace

BALProblem load_bal(const std::string& path) {
    const std::string text = read_all(path);
    Parser p(text);

    BALProblem prob;
    prob.num_cameras = p.next<int>();
    prob.num_points = p.next<int>();
    prob.num_observations = p.next<int>();
    if (prob.num_cameras <= 0 || prob.num_points <= 0 || prob.num_observations <= 0)
        throw std::runtime_error("bad BAL header in " + path);

    prob.obs_camera.resize(prob.num_observations);
    prob.obs_point.resize(prob.num_observations);
    prob.obs_pixel.resize(2 * static_cast<size_t>(prob.num_observations));
    for (int i = 0; i < prob.num_observations; ++i) {
        prob.obs_camera[i] = p.next<int>();
        prob.obs_point[i] = p.next<int>();
        prob.obs_pixel[2 * i] = p.next<double>();
        prob.obs_pixel[2 * i + 1] = p.next<double>();
        if (prob.obs_camera[i] < 0 || prob.obs_camera[i] >= prob.num_cameras ||
            prob.obs_point[i] < 0 || prob.obs_point[i] >= prob.num_points)
            throw std::runtime_error("observation index out of range in " + path);
    }

    prob.cameras.resize(9 * static_cast<size_t>(prob.num_cameras));
    for (auto& v : prob.cameras) v = p.next<double>();

    prob.points.resize(3 * static_cast<size_t>(prob.num_points));
    for (auto& v : prob.points) v = p.next<double>();

    if (!p.at_end()) throw std::runtime_error("trailing garbage in " + path);
    return prob;
}

void snavely_residual(const double* camera9, const double* point3,
                      const double* observed2, double* out2) {
    const double* rod = camera9;
    const double* t = camera9 + 3;
    const double f = camera9[6], k1 = camera9[7], k2 = camera9[8];

    double theta = std::sqrt(rod[0] * rod[0] + rod[1] * rod[1] + rod[2] * rod[2]);
    double P[3];
    if (theta > 1e-12) {
        double w[3] = {rod[0] / theta, rod[1] / theta, rod[2] / theta};
        double c = std::cos(theta), s = std::sin(theta);
        double wx = w[0] * point3[0] + w[1] * point3[1] + w[2] * point3[2];
        double cross[3] = {
            w[1] * point3[2] - w[2] * point3[1],
            w[2] * point3[0] - w[0] * point3[2],
            w[0] * point3[1] - w[1] * point3[0],
        };
        for (int i = 0; i < 3; ++i)
            P[i] = point3[i] * c + cross[i] * s + w[i] * wx * (1.0 - c) + t[i];
    } else {
        for (int i = 0; i < 3; ++i) P[i] = point3[i] + t[i];
    }

    double px = -P[0] / P[2], py = -P[1] / P[2];
    double r2 = px * px + py * py;
    double r = 1.0 + k1 * r2 + k2 * r2 * r2;
    out2[0] = f * r * px - observed2[0];
    out2[1] = f * r * py - observed2[1];
}

double total_squared_error(const BALProblem& p) {
    double sum = 0.0;
    double res[2];
    for (int i = 0; i < p.num_observations; ++i) {
        snavely_residual(p.camera(p.obs_camera[i]), p.point(p.obs_point[i]),
                         &p.obs_pixel[2 * i], res);
        sum += res[0] * res[0] + res[1] * res[1];
    }
    return sum;
}

}  // namespace gba
