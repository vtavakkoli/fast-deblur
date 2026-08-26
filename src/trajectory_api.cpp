#include "internal.hpp"

#include <stdexcept>
#include <string>

namespace fast_deblur {

DeblurResult applyMotionTrajectoryPrior(
    const cv::Mat& image,
    const DeblurResult& baseline,
    const DeblurConfig& config) {
    DeblurConfig cfg = config;
    cfg.validate();
    if (baseline.image.empty() || baseline.kernel.empty()) {
        throw std::invalid_argument("baseline result must contain image and kernel");
    }

    const cv::Mat observed = internal::asFloat01(image);
    if (baseline.image.size() != observed.size()) {
        throw std::invalid_argument("baseline image dimensions must match observed image");
    }

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
        DeblurResult fallback = baseline;
        fallback.restoration_path = "motion-trajectory-rejected/" + baseline.restoration_path;
        return fallback;
    }

    return {
        trajectory_image,
        trajectory_kernel,
        baseline.latent,
        "motion-trajectory/" + trajectory_path,
    };
}

}  // namespace fast_deblur
