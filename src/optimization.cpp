#include "internal.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace fast_deblur::internal {
namespace {

std::pair<cv::Mat, cv::Mat> periodicGradients(const cv::Mat& image) {
    CV_Assert(image.type() == CV_32F);
    cv::Mat gx(image.size(), CV_32F), gy(image.size(), CV_32F);
    for (int y = 0; y < image.rows; ++y) {
        const int yn = (y + 1) % image.rows;
        const float* row = image.ptr<float>(y);
        const float* next = image.ptr<float>(yn);
        float* dx = gx.ptr<float>(y);
        float* dy = gy.ptr<float>(y);
        for (int x = 0; x < image.cols; ++x) {
            const int xn = (x + 1) % image.cols;
            dx[x] = row[xn] - row[x];
            dy[x] = next[x] - row[x];
        }
    }
    return {gx, gy};
}

cv::Mat divergence(const cv::Mat& gx, const cv::Mat& gy) {
    CV_Assert(gx.type() == CV_32F && gy.type() == CV_32F && gx.size() == gy.size());
    cv::Mat div(gx.size(), CV_32F);
    for (int y = 0; y < gx.rows; ++y) {
        const int yp = (y - 1 + gx.rows) % gx.rows;
        const float* dx = gx.ptr<float>(y);
        const float* dyp = gy.ptr<float>(yp);
        const float* dy = gy.ptr<float>(y);
        float* out = div.ptr<float>(y);
        for (int x = 0; x < gx.cols; ++x) {
            const int xp = (x - 1 + gx.cols) % gx.cols;
            out[x] = (dx[xp] - dx[x]) + (dyp[x] - dy[x]);
        }
    }
    return div;
}

cv::Mat gradientDenominator(cv::Size shape) {
    cv::Mat fx = (cv::Mat_<float>(1, 2) << 1.0F, -1.0F);
    cv::Mat fy = (cv::Mat_<float>(2, 1) << 1.0F, -1.0F);
    return abs2(psf2otf(fx, shape)) + abs2(psf2otf(fy, shape));
}

float otsuThreshold01(const cv::Mat& values) {
    CV_Assert(values.type() == CV_32F);
    cv::Mat clipped;
    cv::min(values, 1.0, clipped);
    cv::max(clipped, 0.0, clipped);
    cv::Mat u8;
    clipped.convertTo(u8, CV_8U, 255.0, 0.0);
    cv::Mat tmp;
    const double threshold = cv::threshold(u8, tmp, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
    return std::max(static_cast<float>(threshold / 255.0), 1.0F / 255.0F);
}

cv::Mat solveFrequency(const cv::Mat& norm1, const cv::Mat& den_kernel, const cv::Mat& den_grad,
                       const cv::Mat& div, float beta, const cv::Mat* u = nullptr,
                       float beta_pixel = 0.0F) {
    cv::Mat numerator = norm1.clone();
    numerator += complexScale(fftReal(div), beta);
    if (u && beta_pixel > 0) numerator += complexScale(fftReal(*u), beta_pixel);
    cv::Mat denominator = den_kernel + beta * den_grad;
    if (beta_pixel > 0) denominator += beta_pixel;
    return ifftReal(complexDivideReal(numerator, denominator));
}

cv::Mat l0SingleChannel(const cv::Mat& blurred, const cv::Mat& kernel, float lambda_l0,
                        const DeblurConfig& cfg) {
    cv::Mat s = blurred.clone();
    const cv::Mat ker = psf2otf(kernel, s.size());
    const cv::Mat den_kernel = abs2(ker);
    const cv::Mat den_grad = gradientDenominator(s.size());
    cv::Mat norm1;
    cv::mulSpectrums(fftReal(s), ker, norm1, 0, true);

    float beta = std::max(2.0F * lambda_l0, 1e-8F);
    int steps = 0;
    while (beta < cfg.beta_max_grad) {
        if (cfg.max_grad_steps >= 0 && steps >= cfg.max_grad_steps) break;
        auto [gx, gy] = periodicGradients(s);
        for (int y = 0; y < s.rows; ++y) {
            float* dx = gx.ptr<float>(y);
            float* dy = gy.ptr<float>(y);
            for (int x = 0; x < s.cols; ++x) {
                const float e = dx[x] * dx[x] + dy[x] * dy[x];
                if (e < lambda_l0 / beta) dx[x] = dy[x] = 0.0F;
            }
        }
        s = solveFrequency(norm1, den_kernel, den_grad, divergence(gx, gy), beta);
        beta *= cfg.kappa;
        ++steps;
    }
    return s;
}

cv::Mat tvSingleChannel(const cv::Mat& blurred, const cv::Mat& kernel, float lambda_tv,
                        const DeblurConfig& cfg) {
    cv::Mat result = blurred.clone();
    const cv::Mat ker = psf2otf(kernel, result.size());
    const cv::Mat den1 = abs2(ker);
    const cv::Mat den2 = gradientDenominator(result.size());
    cv::Mat norm1;
    cv::mulSpectrums(fftReal(blurred), ker, norm1, 0, true);

    auto [gx, gy] = periodicGradients(result);
    float beta = 1.0F / std::max(lambda_tv, 1e-12F);
    int steps = 0;
    while (beta > 1e-3F) {
        if (cfg.max_grad_steps >= 0 && steps >= cfg.max_grad_steps) break;
        const float gamma = 1.0F / (2.0F * beta);
        cv::Mat wx(gx.size(), CV_32F), wy(gy.size(), CV_32F);
        for (int y = 0; y < gx.rows; ++y) {
            const float* dx = gx.ptr<float>(y);
            const float* dy = gy.ptr<float>(y);
            float* xout = wx.ptr<float>(y);
            float* yout = wy.ptr<float>(y);
            for (int x = 0; x < gx.cols; ++x) {
                const float sx = std::max(std::abs(dx[x]) - beta * lambda_tv, 0.0F);
                const float sy = std::max(std::abs(dy[x]) - beta * lambda_tv, 0.0F);
                xout[x] = std::copysign(sx, dx[x]);
                yout[x] = std::copysign(sy, dy[x]);
            }
        }
        cv::Mat numerator = norm1 + complexScale(fftReal(divergence(wx, wy)), gamma);
        cv::Mat denominator = den1 + gamma * den2;
        result = ifftReal(complexDivideReal(numerator, denominator));
        std::tie(gx, gy) = periodicGradients(result);
        beta /= 2.0F;
        ++steps;
    }
    return result;
}

std::vector<cv::Mat> splitPlanes(const cv::Mat& image) {
    std::vector<cv::Mat> planes;
    if (image.channels() == 1) planes.push_back(image);
    else cv::split(image, planes);
    return planes;
}

cv::Mat mergePlanes(const std::vector<cv::Mat>& planes) {
    if (planes.size() == 1) return planes[0];
    cv::Mat out;
    cv::merge(planes, out);
    return out;
}

}  // namespace

cv::Mat l0DeblurDarkChannel(const cv::Mat& blurred, const cv::Mat& kernel,
                            float lambda_dark, float lambda_grad, const DeblurConfig& cfg) {
    cv::Mat s = asFloat01(blurred);
    if (s.channels() != 1) throw std::invalid_argument("blind latent optimization expects grayscale input");
    const cv::Mat ker = psf2otf(kernel, s.size());
    const cv::Mat den_kernel = abs2(ker);
    const cv::Mat den_grad = gradientDenominator(s.size());
    cv::Mat norm1;
    cv::mulSpectrums(fftReal(s), ker, norm1, 0, true);

    cv::Mat s2 = s.mul(s);
    float beta_pixel = lambda_dark / otsuThreshold01(s2);
    int dark_steps = 0;
    while (beta_pixel < cfg.beta_max_pixel) {
        if (cfg.max_dark_steps >= 0 && dark_steps >= cfg.max_dark_steps) break;
        cv::Mat u = projectDarkChannel(s, lambda_dark, beta_pixel, cfg.dark_patch_size);
        float beta = std::max(2.0F * lambda_grad, 1e-8F);
        int grad_steps = 0;
        while (beta < cfg.beta_max_grad) {
            if (cfg.max_grad_steps >= 0 && grad_steps >= cfg.max_grad_steps) break;
            auto [gx, gy] = periodicGradients(s);
            for (int y = 0; y < s.rows; ++y) {
                float* dx = gx.ptr<float>(y); float* dy = gy.ptr<float>(y);
                for (int x = 0; x < s.cols; ++x) {
                    const float energy = dx[x] * dx[x] + dy[x] * dy[x];
                    if (energy < lambda_grad / beta) dx[x] = dy[x] = 0.0F;
                }
            }
            s = solveFrequency(norm1, den_kernel, den_grad, divergence(gx, gy), beta, &u, beta_pixel);
            beta *= cfg.kappa; ++grad_steps; if (lambda_grad == 0.0F) break;
        }
        beta_pixel *= cfg.kappa; ++dark_steps;
    }
    return s;
}

cv::Mat l0Restoration(const cv::Mat& blurred, const cv::Mat& kernel,
                      float lambda_l0, const DeblurConfig& cfg, bool pad) {
    cv::Mat image = asFloat01(blurred);
    const cv::Size original = image.size();
    if (pad) image = wrapBoundary(image, fastShape(original, kernel.size()));
    auto planes = splitPlanes(image);
    for (auto& plane : planes) plane = l0SingleChannel(plane, kernel, lambda_l0, cfg);
    cv::Mat out = mergePlanes(planes);
    return out(cv::Rect(0, 0, original.width, original.height)).clone();
}

cv::Mat tvDeconvolutionAniso(const cv::Mat& blurred, const cv::Mat& kernel,
                             float lambda_tv, const DeblurConfig& cfg) {
    cv::Mat image = asFloat01(blurred);
    auto planes = splitPlanes(image);
    for (auto& plane : planes) plane = tvSingleChannel(plane, kernel, lambda_tv, cfg);
    return mergePlanes(planes);
}

cv::Mat ringingArtifactsRemoval(const cv::Mat& image, const cv::Mat& kernel,
                                const DeblurConfig& cfg) {
    const cv::Size original = image.size();
    cv::Mat padded = wrapBoundary(asFloat01(image), fastShape(original, kernel.size()));
    cv::Mat latent_tv = tvDeconvolutionAniso(padded, kernel, cfg.lambda_tv, cfg);
    latent_tv = latent_tv(cv::Rect(0, 0, original.width, original.height)).clone();
    if (cfg.weight_ring == 0.0F) {
        cv::max(latent_tv, 0.0, latent_tv); cv::min(latent_tv, 1.0, latent_tv); return latent_tv;
    }
    cv::Mat latent_l0 = l0Restoration(padded, kernel, cfg.lambda_l0, cfg, false);
    latent_l0 = latent_l0(cv::Rect(0, 0, original.width, original.height)).clone();
    cv::Mat diff = latent_tv - latent_l0, filtered;
    cv::bilateralFilter(diff, filtered, 0, 0.1, 3.0);
    cv::Mat out = latent_tv - cfg.weight_ring * filtered;
    cv::max(out, 0.0, out); cv::min(out, 1.0, out);
    return out;
}

}  // namespace fast_deblur::internal
