#pragma once

#include "fast_deblur/config.hpp"
#include "fast_deblur/fast_deblur.hpp"

#include <opencv2/core.hpp>

#include <optional>
#include <tuple>
#include <utility>
#include <vector>

namespace fast_deblur::internal {

cv::Mat asFloat01(const cv::Mat& image);
cv::Mat matlabRgb2Gray(const cv::Mat& image);
cv::Mat fftReal(const cv::Mat& image);
cv::Mat ifftReal(const cv::Mat& spectrum);
cv::Mat psf2otf(const cv::Mat& psf, cv::Size shape);
cv::Mat otf2psf(const cv::Mat& otf, cv::Size psf_shape);
cv::Mat complexTimesReal(const cv::Mat& complex, const cv::Mat& real);
cv::Mat complexAdd(const cv::Mat& a, const cv::Mat& b);
cv::Mat complexScale(const cv::Mat& a, float scale);
cv::Mat complexDivideReal(const cv::Mat& complex, const cv::Mat& real);
cv::Mat abs2(const cv::Mat& spectrum);
cv::Size fastShape(cv::Size image_shape, cv::Size kernel_shape);

cv::Mat wrapBoundary(const cv::Mat& image, cv::Size target_shape);
cv::Mat darkChannel(const cv::Mat& image, int patch_size);
cv::Mat projectDarkChannel(const cv::Mat& image, float lambda_dark, float beta_pixel, int patch_size);

std::pair<cv::Mat, cv::Mat> validGradients(const cv::Mat& image);
std::tuple<cv::Mat, cv::Mat, double> thresholdGradients(
    const cv::Mat& latent, int psf_size, std::optional<double> threshold);
cv::Mat estimatePsf(const cv::Mat& blurred_x, const cv::Mat& blurred_y,
                    const cv::Mat& latent_x, const cv::Mat& latent_y,
                    float weight, cv::Size psf_shape, float peak_fraction = 0.05F,
                    int max_iter = 20, double tol = 1e-5);
cv::Mat pruneKernel(const cv::Mat& kernel, float min_component_mass = 0.1F);
cv::Mat adjustPsfCenter(const cv::Mat& kernel);
cv::Mat initKernel(int size);
cv::Mat resizeKernel(const cv::Mat& kernel, double scale, int target_size);
cv::Mat refinePsfStructure(const cv::Mat& kernel);

cv::Mat downsampleLevin(const cv::Mat& image, double ratio);
cv::Mat l0DeblurDarkChannel(const cv::Mat& blurred, const cv::Mat& kernel,
                            float lambda_dark, float lambda_grad, const DeblurConfig& cfg);
cv::Mat l0Restoration(const cv::Mat& blurred, const cv::Mat& kernel,
                      float lambda_l0, const DeblurConfig& cfg, bool pad = true);
cv::Mat tvDeconvolutionAniso(const cv::Mat& blurred, const cv::Mat& kernel,
                             float lambda_tv, const DeblurConfig& cfg);
cv::Mat ringingArtifactsRemoval(const cv::Mat& image, const cv::Mat& kernel,
                                const DeblurConfig& cfg);
cv::Mat whyteDeconvolution(const cv::Mat& image, const cv::Mat& kernel,
                           int iterations, int threads = 0);

ArtifactDiagnostics diagnostics(const cv::Mat& observed, const cv::Mat& candidate);
bool rippleRisk(const ArtifactDiagnostics& diag, int kernel_size);
double restorationScore(const cv::Mat& observed, const cv::Mat& candidate,
                        const cv::Mat& reblurred, ArtifactDiagnostics* diag = nullptr);
bool shouldRetryKernel(const cv::Mat& observed, const cv::Mat& restored,
                       const cv::Mat& kernel, int kernel_size, double blind_score);

cv::Mat reblurImage(const cv::Mat& image, const cv::Mat& kernel);
cv::Mat annealedPnpRefine(const cv::Mat& observed, const cv::Mat& initial,
                          const cv::Mat& kernel, int steps = 4,
                          float sigma_start = 0.025F, float sigma_end = 0.004F,
                          int candidates = 2, unsigned seed = 0);
cv::Mat extremeChannelRefine(const cv::Mat& observed, const cv::Mat& initial,
                             const cv::Mat& kernel, int steps = 3, int patch_size = 15);

std::pair<cv::Mat, cv::Mat> estimateBlurKernel(const cv::Mat& gray, const DeblurConfig& cfg);
std::tuple<cv::Mat, double, std::string> restoreAndScore(
    const cv::Mat& image, const cv::Mat& kernel, const DeblurConfig& cfg);

}  // namespace fast_deblur::internal
