#include "fast_deblur/fast_deblur.hpp"
#include "internal.hpp"

#include <opencv2/imgproc.hpp>

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

void check(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

bool finiteMat(const cv::Mat& matrix) {
    cv::Mat flat = matrix.reshape(1);
    for (int y = 0; y < flat.rows; ++y) {
        const float* row = flat.ptr<float>(y);
        for (int x = 0; x < flat.cols; ++x) {
            if (!std::isfinite(row[x])) return false;
        }
    }
    return true;
}

void testPsfRoundTrip() {
    cv::Mat psf = cv::Mat::zeros(7, 7, CV_32F);
    psf.at<float>(2, 2) = .15F;
    psf.at<float>(3, 3) = .55F;
    psf.at<float>(4, 4) = .30F;
    cv::Mat restored = fast_deblur::internal::otf2psf(
        fast_deblur::internal::psf2otf(psf, cv::Size(64, 64)), psf.size());
    check(cv::norm(psf, restored, cv::NORM_INF) < 1e-4, "psf round trip failed");
}

void testKernelInitialization() {
    cv::Mat kernel = fast_deblur::internal::initKernel(25);
    check(std::abs(cv::sum(kernel)[0] - 1.0) < 1e-6, "kernel not normalized");
    check(
        std::abs(kernel.at<float>(11, 11) - .5F) < 1e-6F &&
            std::abs(kernel.at<float>(11, 12) - .5F) < 1e-6F,
        "legacy taps wrong");
}

void testChoFastShape() {
    cv::Size out = fast_deblur::internal::fastShape(cv::Size(257, 193), cv::Size(25, 25));
    check(out.width >= 281 && out.height >= 217, "fast FFT shape too small");
}

void testBoundaryInvariant() {
    cv::Mat source(11, 13, CV_32F);
    cv::randu(source, 0.0F, 1.0F);
    cv::Mat wrapped = fast_deblur::internal::wrapBoundary(source, cv::Size(24, 20));
    check(wrapped.rows == 20 && wrapped.cols == 24, "wrapped shape mismatch");
    check(
        cv::norm(source, wrapped(cv::Rect(0, 0, source.cols, source.rows)), cv::NORM_INF) < 1e-5,
        "wrapping modified source");
    check(finiteMat(wrapped), "non-finite boundary output");
}

void testSequentialDarkProjectionBorder() {
    cv::Mat source(13, 13, CV_32F);
    cv::randu(source, .1F, .9F);
    cv::Mat projected = fast_deblur::internal::projectDarkChannel(source, .004F, .05F, 5);
    check(projected.size() == source.size(), "dark shape mismatch");
    check(
        cv::norm(
            source(cv::Rect(0, 0, source.cols, 2)),
            projected(cv::Rect(0, 0, source.cols, 2)),
            cv::NORM_INF) < 1e-6,
        "dark border changed");
    check(finiteMat(projected), "non-finite dark output");
}

double curveMass(const cv::Mat& kernel, bool on_curve) {
    double mass = 0.0;
    for (int y = 0; y < kernel.rows; ++y) {
        const float* row = kernel.ptr<float>(y);
        for (int x = 0; x < kernel.cols; ++x) {
            const double dx = static_cast<double>(x - kernel.cols / 2);
            const double expected_y = kernel.rows / 2.0 + 0.010 * dx * dx - 4.0;
            const bool near = std::abs(static_cast<double>(y) - expected_y) <= 3.0;
            if (near == on_curve) mass += row[x];
        }
    }
    return mass;
}

void testMotionTrajectoryPriorSuppressesOffPathMass() {
    constexpr int size = 65;
    cv::Mat kernel = cv::Mat::zeros(size, size, CV_32F);
    cv::Point previous(8, 34);
    for (int x = 9; x <= 56; ++x) {
        const double dx = static_cast<double>(x - size / 2);
        const int y = static_cast<int>(std::lround(size / 2.0 + 0.010 * dx * dx - 4.0));
        const cv::Point current(x, y);
        cv::line(kernel, previous, current, cv::Scalar(1.0F), 2, cv::LINE_AA);
        previous = current;
    }
    cv::line(kernel, cv::Point(4, 10), cv::Point(60, 10), cv::Scalar(0.20F), 1, cv::LINE_AA);
    cv::circle(kernel, cv::Point(55, 52), 3, cv::Scalar(0.35F), cv::FILLED, cv::LINE_AA);
    cv::GaussianBlur(kernel, kernel, cv::Size(), 0.7, 0.7, cv::BORDER_REFLECT_101);
    kernel /= static_cast<float>(cv::sum(kernel)[0]);

    const double before_off_path = curveMass(kernel, false);
    const cv::Mat cleaned = fast_deblur::internal::trajectoryRegularizePsf(kernel);
    const double after_off_path = curveMass(cleaned, false);

    check(cleaned.size() == kernel.size(), "trajectory prior changed PSF support");
    check(finiteMat(cleaned), "trajectory prior produced non-finite PSF");
    check(std::abs(cv::sum(cleaned)[0] - 1.0) < 1e-5, "trajectory PSF not normalized");
    check(after_off_path < before_off_path * 0.70, "trajectory prior did not suppress off-path mass");
    check(curveMass(cleaned, true) > 0.70, "trajectory prior did not preserve the motion curve");
}

void testSyntheticSmoke() {
    cv::Mat sharp = cv::Mat::zeros(72, 96, CV_32FC3);
    cv::rectangle(sharp, cv::Rect(12, 15, 28, 32), cv::Scalar(.85, .25, .15), cv::FILLED);
    cv::line(sharp, cv::Point(48, 12), cv::Point(82, 58), cv::Scalar(.2, .9, .4), 4, cv::LINE_AA);
    cv::Mat known = cv::Mat::zeros(9, 9, CV_32F);
    for (int i = 2; i <= 6; ++i) known.at<float>(4, i) = .2F;
    cv::Mat observed = fast_deblur::reblur(sharp, known);

    fast_deblur::DeblurConfig cfg;
    cfg.kernel_size = 9;
    cfg.dark_patch_size = 7;
    cfg.xk_iter = 1;
    cfg.max_grad_steps = 2;
    cfg.max_dark_steps = 1;
    cfg.robust_selection = false;
    cfg.retry_gradient_only = false;
    cfg.conservative_restoration = false;
    cfg.lambda_tv = .002F;
    cfg.lambda_l0 = .001F;

    auto result = fast_deblur::deblur(observed, cfg, fast_deblur::Method::Baseline);
    check(result.image.size() == observed.size(), "deblur shape mismatch");
    check(result.kernel.rows == 9 && result.kernel.cols == 9, "kernel shape mismatch");
    check(std::abs(cv::sum(result.kernel)[0] - 1.0) < 5e-3, "estimated kernel not normalized");
    check(finiteMat(result.image) && finiteMat(result.kernel), "non-finite deblur");

    cv::Mat pnp = fast_deblur::refine(
        observed, result.image, result.kernel, fast_deblur::Method::AnnealedPnP);
    cv::Mat extreme = fast_deblur::refine(
        observed, result.image, result.kernel, fast_deblur::Method::ExtremeChannel);
    check(pnp.size() == observed.size() && extreme.size() == observed.size(), "refinement shape mismatch");
    check(finiteMat(pnp) && finiteMat(extreme), "non-finite refinement");

    auto trajectory = fast_deblur::deblurMotionTrajectory(observed, cfg);
    check(trajectory.image.size() == observed.size(), "trajectory deblur shape mismatch");
    check(trajectory.kernel.size() == result.kernel.size(), "trajectory kernel shape mismatch");
    check(finiteMat(trajectory.image) && finiteMat(trajectory.kernel), "non-finite trajectory result");
    check(std::abs(cv::sum(trajectory.kernel)[0] - 1.0) < 5e-3, "trajectory kernel not normalized");
}

}  // namespace

int main() {
    try {
        testPsfRoundTrip();
        testKernelInitialization();
        testChoFastShape();
        testBoundaryInvariant();
        testSequentialDarkProjectionBorder();
        testMotionTrajectoryPriorSuppressesOffPathMass();
        testSyntheticSmoke();
        std::cout << "All fast-deblur C++ tests passed.\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "TEST FAILURE: " << error.what() << '\n';
        return 1;
    }
}
