#include "internal.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace fast_deblur::internal {
namespace {

struct WeightedPoint {
    cv::Point2d point;
    double weight = 0.0;
    double mass = 0.0;
};

cv::Mat normalizeNonnegative(const cv::Mat& input) {
    cv::Mat out;
    cv::max(input, 0.0, out);
    const double total = cv::sum(out)[0];
    if (total > 0.0) out /= static_cast<float>(total);
    return out;
}

cv::Mat dominantSupport(const cv::Mat& kernel, float peak_fraction) {
    double peak = 0.0;
    cv::minMaxLoc(kernel, nullptr, &peak);
    if (peak <= 0.0) return cv::Mat::zeros(kernel.size(), CV_8U);

    cv::Mat mask;
    cv::compare(kernel, peak * peak_fraction, mask, cv::CMP_GE);
    cv::morphologyEx(
        mask,
        mask,
        cv::MORPH_CLOSE,
        cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3)));

    cv::Mat labels, stats, centroids;
    const int count = cv::connectedComponentsWithStats(mask, labels, stats, centroids, 8, CV_32S);
    if (count <= 1) return mask;

    int best_label = 1;
    double best_mass = -1.0;
    for (int label = 1; label < count; ++label) {
        double mass = 0.0;
        for (int y = 0; y < kernel.rows; ++y) {
            const int* lrow = labels.ptr<int>(y);
            const float* krow = kernel.ptr<float>(y);
            for (int x = 0; x < kernel.cols; ++x) {
                if (lrow[x] == label) mass += krow[x];
            }
        }
        if (mass > best_mass) {
            best_mass = mass;
            best_label = label;
        }
    }

    cv::Mat dominant = cv::Mat::zeros(kernel.size(), CV_8U);
    for (int y = 0; y < kernel.rows; ++y) {
        const int* lrow = labels.ptr<int>(y);
        uchar* drow = dominant.ptr<uchar>(y);
        for (int x = 0; x < kernel.cols; ++x) {
            if (lrow[x] == best_label) drow[x] = 255;
        }
    }
    return dominant;
}

std::vector<WeightedPoint> supportPoints(const cv::Mat& kernel, const cv::Mat& mask) {
    double peak = 0.0;
    cv::minMaxLoc(kernel, nullptr, &peak);
    std::vector<WeightedPoint> points;
    for (int y = 0; y < kernel.rows; ++y) {
        const float* krow = kernel.ptr<float>(y);
        const uchar* mrow = mask.ptr<uchar>(y);
        for (int x = 0; x < kernel.cols; ++x) {
            if (mrow[x] == 0 || krow[x] <= 0.0F) continue;
            const double normalized = krow[x] / std::max(peak, 1e-12);
            points.push_back({
                cv::Point2d(static_cast<double>(x), static_cast<double>(y)),
                std::pow(normalized, 1.55),
                static_cast<double>(krow[x]),
            });
        }
    }
    return points;
}

struct AxisModel {
    cv::Point2d center;
    cv::Point2d major;
    double lambda_major = 0.0;
    double lambda_minor = 0.0;
    double anisotropy = 1.0;
};

AxisModel weightedAxis(const std::vector<WeightedPoint>& points) {
    double sw = 0.0, cx = 0.0, cy = 0.0;
    for (const auto& sample : points) {
        sw += sample.weight;
        cx += sample.weight * sample.point.x;
        cy += sample.weight * sample.point.y;
    }
    if (sw <= 0.0) return {{0.0, 0.0}, {1.0, 0.0}, 0.0, 0.0, 1.0};
    cx /= sw;
    cy /= sw;

    double cxx = 0.0, cxy = 0.0, cyy = 0.0;
    for (const auto& sample : points) {
        const double dx = sample.point.x - cx;
        const double dy = sample.point.y - cy;
        cxx += sample.weight * dx * dx;
        cxy += sample.weight * dx * dy;
        cyy += sample.weight * dy * dy;
    }
    cxx /= sw;
    cxy /= sw;
    cyy /= sw;

    const double trace = cxx + cyy;
    const double disc = std::sqrt(std::max(0.0, (cxx - cyy) * (cxx - cyy) + 4.0 * cxy * cxy));
    const double major_value = 0.5 * (trace + disc);
    const double minor_value = std::max(0.0, 0.5 * (trace - disc));

    cv::Point2d major;
    if (std::abs(cxy) > 1e-12) {
        major = cv::Point2d(major_value - cyy, cxy);
    } else {
        major = cxx >= cyy ? cv::Point2d(1.0, 0.0) : cv::Point2d(0.0, 1.0);
    }
    const double norm = std::hypot(major.x, major.y);
    if (norm > 0.0) {
        major.x /= norm;
        major.y /= norm;
    } else {
        major = cv::Point2d(1.0, 0.0);
    }

    return {
        cv::Point2d(cx, cy),
        major,
        major_value,
        minor_value,
        major_value / std::max(minor_value, 1e-9),
    };
}

std::vector<cv::Point2d> initialControlPoints(
    const std::vector<WeightedPoint>& points,
    const AxisModel& axis,
    int count,
    std::vector<double>* masses) {
    double tmin = std::numeric_limits<double>::infinity();
    double tmax = -std::numeric_limits<double>::infinity();
    std::vector<double> t(points.size(), 0.0);
    for (std::size_t i = 0; i < points.size(); ++i) {
        const double dx = points[i].point.x - axis.center.x;
        const double dy = points[i].point.y - axis.center.y;
        t[i] = dx * axis.major.x + dy * axis.major.y;
        tmin = std::min(tmin, t[i]);
        tmax = std::max(tmax, t[i]);
    }
    if (!(tmax > tmin)) tmax = tmin + 1.0;

    std::vector<cv::Point2d> controls(static_cast<std::size_t>(count));
    masses->assign(static_cast<std::size_t>(count), 0.0);
    std::vector<double> weight_sums(static_cast<std::size_t>(count), 0.0);

    for (std::size_t i = 0; i < points.size(); ++i) {
        const double u = (t[i] - tmin) / (tmax - tmin);
        const int bin = std::clamp(static_cast<int>(std::floor(u * count)), 0, count - 1);
        controls[static_cast<std::size_t>(bin)].x += points[i].weight * points[i].point.x;
        controls[static_cast<std::size_t>(bin)].y += points[i].weight * points[i].point.y;
        weight_sums[static_cast<std::size_t>(bin)] += points[i].weight;
        (*masses)[static_cast<std::size_t>(bin)] += points[i].mass;
    }

    for (int i = 0; i < count; ++i) {
        if (weight_sums[static_cast<std::size_t>(i)] > 0.0) {
            controls[static_cast<std::size_t>(i)].x /= weight_sums[static_cast<std::size_t>(i)];
            controls[static_cast<std::size_t>(i)].y /= weight_sums[static_cast<std::size_t>(i)];
        } else {
            const double u = (static_cast<double>(i) + 0.5) / static_cast<double>(count);
            const double tt = tmin + u * (tmax - tmin);
            controls[static_cast<std::size_t>(i)] = axis.center + tt * axis.major;
        }
    }

    for (int i = 0; i < count; ++i) {
        if ((*masses)[static_cast<std::size_t>(i)] > 0.0) continue;
        int left = i - 1;
        while (left >= 0 && (*masses)[static_cast<std::size_t>(left)] <= 0.0) --left;
        int right = i + 1;
        while (right < count && (*masses)[static_cast<std::size_t>(right)] <= 0.0) ++right;
        if (left >= 0 && right < count) {
            const double u = static_cast<double>(i - left) / static_cast<double>(right - left);
            (*masses)[static_cast<std::size_t>(i)] =
                (1.0 - u) * (*masses)[static_cast<std::size_t>(left)] +
                u * (*masses)[static_cast<std::size_t>(right)];
        } else if (left >= 0) {
            (*masses)[static_cast<std::size_t>(i)] = (*masses)[static_cast<std::size_t>(left)];
        } else if (right < count) {
            (*masses)[static_cast<std::size_t>(i)] = (*masses)[static_cast<std::size_t>(right)];
        }
    }
    return controls;
}

void smoothControls(std::vector<cv::Point2d>* controls) {
    if (controls->size() < 5) return;
    const std::vector<cv::Point2d> source = *controls;
    constexpr double weights[5] = {0.1, 0.2, 0.4, 0.2, 0.1};
    for (std::size_t i = 0; i < controls->size(); ++i) {
        cv::Point2d smoothed(0.0, 0.0);
        for (int k = -2; k <= 2; ++k) {
            const std::size_t j = static_cast<std::size_t>(std::clamp<int>(
                static_cast<int>(i) + k, 0, static_cast<int>(controls->size()) - 1));
            smoothed += weights[k + 2] * source[j];
        }
        (*controls)[i] = 0.55 * source[i] + 0.45 * smoothed;
    }
}

void refinePrincipalCurve(
    const std::vector<WeightedPoint>& points,
    std::vector<cv::Point2d>* controls,
    std::vector<double>* masses) {
    const int count = static_cast<int>(controls->size());
    for (int iteration = 0; iteration < 4; ++iteration) {
        std::vector<cv::Point2d> sums(static_cast<std::size_t>(count), cv::Point2d(0.0, 0.0));
        std::vector<double> weights(static_cast<std::size_t>(count), 0.0);
        std::vector<double> next_masses(static_cast<std::size_t>(count), 0.0);

        for (const auto& sample : points) {
            int best = 0;
            double best_d2 = std::numeric_limits<double>::infinity();
            for (int i = 0; i < count; ++i) {
                const cv::Point2d delta = sample.point - (*controls)[static_cast<std::size_t>(i)];
                const double d2 = delta.dot(delta);
                if (d2 < best_d2) {
                    best_d2 = d2;
                    best = i;
                }
            }
            sums[static_cast<std::size_t>(best)] += sample.weight * sample.point;
            weights[static_cast<std::size_t>(best)] += sample.weight;
            next_masses[static_cast<std::size_t>(best)] += sample.mass;
        }

        for (int i = 0; i < count; ++i) {
            if (weights[static_cast<std::size_t>(i)] > 0.0) {
                (*controls)[static_cast<std::size_t>(i)] =
                    sums[static_cast<std::size_t>(i)] / weights[static_cast<std::size_t>(i)];
            }
            if (next_masses[static_cast<std::size_t>(i)] > 0.0) {
                (*masses)[static_cast<std::size_t>(i)] = next_masses[static_cast<std::size_t>(i)];
            }
        }
        smoothControls(controls);

        const std::vector<double> source = *masses;
        for (int i = 0; i < count; ++i) {
            const int left = std::max(0, i - 1);
            const int right = std::min(count - 1, i + 1);
            (*masses)[static_cast<std::size_t>(i)] =
                0.25 * source[static_cast<std::size_t>(left)] +
                0.50 * source[static_cast<std::size_t>(i)] +
                0.25 * source[static_cast<std::size_t>(right)];
        }
    }
}

double pointSegmentDistanceSquared(const cv::Point2d& p, const cv::Point2d& a, const cv::Point2d& b) {
    const cv::Point2d ab = b - a;
    const double denom = ab.dot(ab);
    if (denom <= 1e-12) {
        const cv::Point2d d = p - a;
        return d.dot(d);
    }
    const cv::Point2d ap = p - a;
    const double t = std::clamp(ap.dot(ab) / denom, 0.0, 1.0);
    const cv::Point2d d = p - (a + t * ab);
    return d.dot(d);
}

cv::Mat distanceToCurve(cv::Size size, const std::vector<cv::Point2d>& controls) {
    cv::Mat distance(size, CV_32F, cv::Scalar(0.0F));
    for (int y = 0; y < size.height; ++y) {
        float* row = distance.ptr<float>(y);
        for (int x = 0; x < size.width; ++x) {
            const cv::Point2d p(static_cast<double>(x), static_cast<double>(y));
            double best = std::numeric_limits<double>::infinity();
            for (std::size_t i = 0; i + 1 < controls.size(); ++i) {
                best = std::min(best, pointSegmentDistanceSquared(p, controls[i], controls[i + 1]));
            }
            row[x] = static_cast<float>(std::sqrt(std::max(0.0, best)));
        }
    }
    return distance;
}

void splat(cv::Mat* image, const cv::Point2d& p, double value) {
    const int x0 = static_cast<int>(std::floor(p.x));
    const int y0 = static_cast<int>(std::floor(p.y));
    const double fx = p.x - x0;
    const double fy = p.y - y0;
    for (int dy = 0; dy <= 1; ++dy) {
        for (int dx = 0; dx <= 1; ++dx) {
            const int x = x0 + dx;
            const int y = y0 + dy;
            if (x < 0 || y < 0 || x >= image->cols || y >= image->rows) continue;
            const double wx = dx == 0 ? 1.0 - fx : fx;
            const double wy = dy == 0 ? 1.0 - fy : fy;
            image->at<float>(y, x) += static_cast<float>(value * wx * wy);
        }
    }
}

cv::Mat projectMassToCurve(
    cv::Size size,
    const std::vector<cv::Point2d>& controls,
    const std::vector<double>& masses) {
    cv::Mat trajectory = cv::Mat::zeros(size, CV_32F);
    for (std::size_t i = 0; i + 1 < controls.size(); ++i) {
        const cv::Point2d a = controls[i];
        const cv::Point2d b = controls[i + 1];
        const double length = cv::norm(b - a);
        const int steps = std::max(2, static_cast<int>(std::ceil(length * 2.0)));
        for (int step = 0; step <= steps; ++step) {
            const double u = static_cast<double>(step) / static_cast<double>(steps);
            const cv::Point2d p = (1.0 - u) * a + u * b;
            const double mass = ((1.0 - u) * masses[i] + u * masses[i + 1]) /
                                static_cast<double>(steps + 1);
            splat(&trajectory, p, mass);
        }
    }
    const double sigma = std::max(0.75, 0.012 * std::max(size.width, size.height));
    cv::GaussianBlur(trajectory, trajectory, cv::Size(), sigma, sigma, cv::BORDER_REFLECT_101);
    return normalizeNonnegative(trajectory);
}

}  // namespace

cv::Mat trajectoryRegularizePsf(const cv::Mat& kernel) {
    cv::Mat k = normalizeNonnegative(kernel);
    double peak = 0.0;
    cv::minMaxLoc(k, nullptr, &peak);
    if (peak <= 0.0 || k.rows < 7 || k.cols < 7) return k;

    const cv::Mat support = dominantSupport(k, 0.02F);
    const std::vector<WeightedPoint> points = supportPoints(k, support);
    if (points.size() < 8) return k;

    const AxisModel axis = weightedAxis(points);
    const int max_dim = std::max(k.rows, k.cols);
    int controls_count = std::clamp(max_dim / 5, 9, 31);
    if (controls_count % 2 == 0) ++controls_count;

    std::vector<double> masses;
    std::vector<cv::Point2d> controls = initialControlPoints(points, axis, controls_count, &masses);
    refinePrincipalCurve(points, &controls, &masses);

    const cv::Mat distance = distanceToCurve(k.size(), controls);
    const double tube_radius = std::max(2.0, 0.035 * max_dim);
    double coverage = 0.0;
    for (int y = 0; y < k.rows; ++y) {
        const float* krow = k.ptr<float>(y);
        const float* drow = distance.ptr<float>(y);
        for (int x = 0; x < k.cols; ++x) {
            if (drow[x] <= tube_radius) coverage += krow[x];
        }
    }

    const double coverage_confidence = std::clamp((coverage - 0.45) / 0.40, 0.0, 1.0);
    const double anisotropy_confidence = std::clamp((axis.anisotropy - 1.0) / 1.5, 0.0, 1.0);
    const double confidence = coverage_confidence * anisotropy_confidence;

    cv::Mat trajectory = projectMassToCurve(k.size(), controls, masses);
    if (cv::sum(trajectory)[0] <= 0.0) return k;

    const double alpha = std::clamp(0.30 + 0.55 * confidence, 0.30, 0.88);
    cv::Mat out = (1.0 - alpha) * k + alpha * trajectory;

    const double sigma = std::max(1.25, 0.025 * max_dim);
    cv::Mat tube(k.size(), CV_32F);
    for (int y = 0; y < k.rows; ++y) {
        const float* drow = distance.ptr<float>(y);
        float* trow = tube.ptr<float>(y);
        for (int x = 0; x < k.cols; ++x) {
            const double d = drow[x];
            trow[x] = static_cast<float>(std::exp(-(d * d) / (2.0 * sigma * sigma)));
        }
    }

    cv::Mat strong;
    cv::compare(k, peak * 0.15, strong, cv::CMP_GE);
    strong.convertTo(strong, CV_32F, 1.0 / 255.0);
    cv::GaussianBlur(strong, strong, cv::Size(), 1.2, 1.2, cv::BORDER_REFLECT_101);
    cv::max(tube, strong, tube);

    out = out.mul(0.08F + 0.92F * tube);
    double out_peak = 0.0;
    cv::minMaxLoc(out, nullptr, &out_peak);
    if (out_peak > 0.0) {
        for (int y = 0; y < out.rows; ++y) {
            float* row = out.ptr<float>(y);
            for (int x = 0; x < out.cols; ++x) {
                if (row[x] < out_peak * 0.008) row[x] = 0.0F;
            }
        }
    }

    out = normalizeNonnegative(out);
    out = refinePsfStructure(out);
    return adjustPsfCenter(out);
}

}  // namespace fast_deblur::internal

namespace fast_deblur {

DeblurResult deblurMotionTrajectory(const cv::Mat& image, const DeblurConfig& config) {
    DeblurConfig cfg = config;
    cfg.validate();
    if (cfg.threads > 0) cv::setNumThreads(cfg.threads);

    const cv::Mat observed = internal::asFloat01(image);
    DeblurResult baseline = deblur(observed, cfg, Method::Baseline);
    const cv::Mat trajectory_kernel = internal::trajectoryRegularizePsf(baseline.kernel);

    auto [trajectory_image, trajectory_score, trajectory_path] =
        internal::restoreAndScore(observed, trajectory_kernel, cfg);

    const double baseline_score = internal::restorationScore(
        observed,
        baseline.image,
        internal::reblurImage(baseline.image, baseline.kernel));
    const ArtifactDiagnostics trajectory_diag = internal::diagnostics(observed, trajectory_image);

    if (internal::rippleRisk(trajectory_diag, cfg.kernel_size) ||
        trajectory_score > baseline_score * 1.20) {
        baseline.restoration_path = "motion-trajectory-rejected/" + baseline.restoration_path;
        return baseline;
    }

    return {
        trajectory_image,
        trajectory_kernel,
        baseline.latent,
        "motion-trajectory/" + trajectory_path,
    };
}

}  // namespace fast_deblur
