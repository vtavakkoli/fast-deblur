#include "internal.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <mutex>
#include <stdexcept>

namespace fast_deblur {

void DeblurConfig::validate() const {
    if (kernel_size < 3 || kernel_size % 2 == 0) throw std::invalid_argument("kernel_size must be odd and >= 3");
    if (dark_patch_size < 3 || dark_patch_size % 2 == 0) throw std::invalid_argument("dark_patch_size must be odd and >= 3");
    if (gamma_correct <= 0) throw std::invalid_argument("gamma_correct must be > 0");
    if (xk_iter < 1) throw std::invalid_argument("xk_iter must be >= 1");
    if (kappa <= 1) throw std::invalid_argument("kappa must be > 1");
    if (prescale <= 0) throw std::invalid_argument("prescale must be > 0");
    if (saturation_iterations < 1) throw std::invalid_argument("saturation_iterations must be >= 1");
}

DeblurConfig DeblurConfig::matlabParity() {
    DeblurConfig cfg;
    cfg.robust_selection = false;
    cfg.retry_gradient_only = false;
    cfg.conservative_restoration = false;
    return cfg;
}

namespace internal {

cv::Mat asFloat01(const cv::Mat& image) {
    if (image.empty()) throw std::invalid_argument("image is empty");
    cv::Mat out;
    if (image.depth() == CV_32F) {
        out = image.clone();
    } else if (image.depth() == CV_64F) {
        image.convertTo(out, CV_MAKETYPE(CV_32F, image.channels()));
    } else {
        double scale = 1.0;
        if (image.depth() == CV_8U) scale = 1.0 / 255.0;
        else if (image.depth() == CV_16U) scale = 1.0 / 65535.0;
        image.convertTo(out, CV_MAKETYPE(CV_32F, image.channels()), scale);
    }
    return out;
}

cv::Mat matlabRgb2Gray(const cv::Mat& image) {
    if (image.channels() == 1) return image.clone();
    if (image.channels() != 3) throw std::invalid_argument("matlabRgb2Gray expects 1 or 3 channels");
    std::vector<cv::Mat> c;
    cv::split(image, c);
    cv::Mat gray = c[0] * 0.298936021293775F + c[1] * 0.587043074451121F + c[2] * 0.114020904255103F;
    return gray;
}

cv::Mat fftReal(const cv::Mat& image) {
    CV_Assert(image.channels() == 1 && image.depth() == CV_32F);
    cv::Mat out;
    cv::dft(image, out, cv::DFT_COMPLEX_OUTPUT);
    return out;
}

cv::Mat ifftReal(const cv::Mat& spectrum) {
    CV_Assert(spectrum.type() == CV_32FC2);
    cv::Mat out;
    cv::dft(spectrum, out, cv::DFT_INVERSE | cv::DFT_SCALE | cv::DFT_REAL_OUTPUT);
    return out;
}

static cv::Mat circularShift(const cv::Mat& src, int dy, int dx) {
    CV_Assert(src.channels() == 1);
    cv::Mat dst(src.size(), src.type());
    const int h = src.rows;
    const int w = src.cols;
    dy = ((dy % h) + h) % h;
    dx = ((dx % w) + w) % w;
    for (int y = 0; y < h; ++y) {
        int sy = (y - dy + h) % h;
        const float* s = src.ptr<float>(sy);
        float* d = dst.ptr<float>(y);
        for (int x = 0; x < w; ++x) d[x] = s[(x - dx + w) % w];
    }
    return dst;
}

cv::Mat psf2otf(const cv::Mat& psf, cv::Size shape) {
    CV_Assert(psf.channels() == 1 && psf.depth() == CV_32F);
    if (psf.rows > shape.height || psf.cols > shape.width) throw std::invalid_argument("PSF larger than OTF shape");
    cv::Mat padded = cv::Mat::zeros(shape, CV_32F);
    psf.copyTo(padded(cv::Rect(0, 0, psf.cols, psf.rows)));
    padded = circularShift(padded, -(psf.rows / 2), -(psf.cols / 2));
    return fftReal(padded);
}

cv::Mat otf2psf(const cv::Mat& otf, cv::Size psf_shape) {
    cv::Mat spatial = ifftReal(otf);
    spatial = circularShift(spatial, psf_shape.height / 2, psf_shape.width / 2);
    return spatial(cv::Rect(0, 0, psf_shape.width, psf_shape.height)).clone();
}

cv::Mat complexTimesReal(const cv::Mat& complex, const cv::Mat& real) {
    CV_Assert(complex.type() == CV_32FC2 && real.type() == CV_32F && complex.size() == real.size());
    cv::Mat out(complex.size(), CV_32FC2);
    for (int y = 0; y < complex.rows; ++y) {
        const cv::Vec2f* c = complex.ptr<cv::Vec2f>(y);
        const float* r = real.ptr<float>(y);
        cv::Vec2f* d = out.ptr<cv::Vec2f>(y);
        for (int x = 0; x < complex.cols; ++x) d[x] = c[x] * r[x];
    }
    return out;
}

cv::Mat complexAdd(const cv::Mat& a, const cv::Mat& b) {
    cv::Mat out;
    cv::add(a, b, out);
    return out;
}

cv::Mat complexScale(const cv::Mat& a, float scale) {
    cv::Mat out;
    a.convertTo(out, a.type(), scale);
    return out;
}

cv::Mat complexDivideReal(const cv::Mat& complex, const cv::Mat& real) {
    CV_Assert(complex.type() == CV_32FC2 && real.type() == CV_32F && complex.size() == real.size());
    cv::Mat out(complex.size(), CV_32FC2);
    for (int y = 0; y < complex.rows; ++y) {
        const cv::Vec2f* c = complex.ptr<cv::Vec2f>(y);
        const float* r = real.ptr<float>(y);
        cv::Vec2f* d = out.ptr<cv::Vec2f>(y);
        for (int x = 0; x < complex.cols; ++x) {
            float denom = std::max(r[x], 1e-20F);
            d[x] = c[x] / denom;
        }
    }
    return out;
}

cv::Mat abs2(const cv::Mat& spectrum) {
    CV_Assert(spectrum.type() == CV_32FC2);
    cv::Mat out(spectrum.size(), CV_32F);
    for (int y = 0; y < spectrum.rows; ++y) {
        const cv::Vec2f* s = spectrum.ptr<cv::Vec2f>(y);
        float* d = out.ptr<float>(y);
        for (int x = 0; x < spectrum.cols; ++x) d[x] = s[x][0] * s[x][0] + s[x][1] * s[x][1];
    }
    return out;
}

static int choFftSize(int value) {
    if (value < 1) throw std::invalid_argument("FFT dimension must be positive");
    if (value > 4096) return cv::getOptimalDFTSize(value);
    static std::once_flag once;
    static std::vector<int> lut(4097, 0);
    std::call_once(once, [] {
        const int limit = 4096;
        for (long long e2 = 1; e2 <= limit; e2 *= 2)
            for (long long e3 = e2; e3 <= limit; e3 *= 3)
                for (long long e5 = e3; e5 <= limit; e5 *= 5)
                    for (long long e7 = e5; e7 <= limit; e7 *= 7) {
                        lut[static_cast<int>(e7)] = static_cast<int>(e7);
                        if (e7 * 11 <= limit) lut[static_cast<int>(e7 * 11)] = static_cast<int>(e7 * 11);
                        if (e7 * 13 <= limit) lut[static_cast<int>(e7 * 13)] = static_cast<int>(e7 * 13);
                        if (e7 > limit / 7) break;
                    }
        int next = 0;
        for (int i = limit; i >= 1; --i) {
            if (lut[i] != 0) next = i;
            else lut[i] = next;
        }
    });
    return lut[value] > 0 ? lut[value] : cv::getOptimalDFTSize(value);
}

cv::Size fastShape(cv::Size image_shape, cv::Size kernel_shape) {
    return cv::Size(
        choFftSize(image_shape.width + kernel_shape.width - 1),
        choFftSize(image_shape.height + kernel_shape.height - 1));
}

}  // namespace internal
}  // namespace fast_deblur
