#pragma once

#include "fast_deblur/config.hpp"

#include <opencv2/core.hpp>

#include <string>

namespace fast_deblur {

struct DeblurResult {
    cv::Mat image;   // CV_32F/CV_32FC3, clipped to [0,1]
    cv::Mat kernel;  // CV_32F, normalized to unit mass
    cv::Mat latent;  // last grayscale latent estimate
    std::string restoration_path;
};

struct ArtifactDiagnostics {
    double edge_ratio = 1.0;
    double noise_ratio = 1.0;
    double highpass_ratio = 1.0;
    double clipping_growth = 0.0;
};

DeblurResult deblur(const cv::Mat& image, const DeblurConfig& config = {}, Method method = Method::Baseline);
DeblurResult deblurMotionTrajectory(const cv::Mat& image, const DeblurConfig& config = {});
DeblurResult applyMotionTrajectoryPrior(
    const cv::Mat& image,
    const DeblurResult& baseline,
    const DeblurConfig& config = {});
cv::Mat refine(const cv::Mat& observed, const cv::Mat& baseline, const cv::Mat& kernel, Method method);
cv::Mat reblur(const cv::Mat& image, const cv::Mat& kernel);
ArtifactDiagnostics artifactDiagnostics(const cv::Mat& observed, const cv::Mat& candidate);
double rmse(const cv::Mat& a, const cv::Mat& b);
double psnr(const cv::Mat& reference, const cv::Mat& candidate);
double ssim(const cv::Mat& reference, const cv::Mat& candidate);

}  // namespace fast_deblur
