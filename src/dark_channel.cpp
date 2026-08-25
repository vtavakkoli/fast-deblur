#include "internal.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>
#include <vector>

namespace fast_deblur::internal {

cv::Mat darkChannel(const cv::Mat& image, int patch_size) {
    if (patch_size < 1 || patch_size % 2 == 0) throw std::invalid_argument("patch_size must be positive and odd");
    cv::Mat arr = asFloat01(image);
    cv::Mat base;
    if (arr.channels() == 1) {
        base = arr;
    } else {
        std::vector<cv::Mat> planes;
        cv::split(arr, planes);
        base = planes[0].clone();
        for (std::size_t i = 1; i < planes.size(); ++i) cv::min(base, planes[i], base);
    }
    cv::Mat out;
    cv::Mat kernel = cv::Mat::ones(patch_size, patch_size, CV_8U);
    cv::erode(base, out, kernel, cv::Point(-1, -1), 1, cv::BORDER_REPLICATE);
    return out;
}

cv::Mat projectDarkChannel(const cv::Mat& image, float lambda_dark, float beta_pixel, int patch_size) {
    if (beta_pixel <= 0) throw std::invalid_argument("beta_pixel must be > 0");
    cv::Mat source = asFloat01(image);
    if (source.channels() != 1) throw std::invalid_argument("projectDarkChannel expects grayscale input");

    const cv::Mat minima = darkChannel(source, patch_size);
    cv::Mat selected(source.size(), CV_8U);
    const float threshold = lambda_dark / beta_pixel;
    for (int y = 0; y < source.rows; ++y) {
        const float* m = minima.ptr<float>(y);
        uchar* s = selected.ptr<uchar>(y);
        for (int x = 0; x < source.cols; ++x) s[x] = (m[x] * m[x] < threshold) ? 1 : 0;
    }

    const int radius = patch_size / 2;
    cv::Mat padded;
    cv::copyMakeBorder(source, padded, radius, radius, radius, radius, cv::BORDER_REPLICATE);

    struct Target { int y; int x; };
    std::vector<Target> targets(static_cast<std::size_t>(source.rows) * source.cols);

    cv::parallel_for_(cv::Range(0, source.rows * source.cols), [&](const cv::Range& range) {
        for (int index = range.start; index < range.end; ++index) {
            const int y = index / source.cols;
            const int x = index - y * source.cols;
            const float target = minima.at<float>(y, x);
            Target t{y, x};
            bool found = false;
            for (int xx = 0; xx < patch_size && !found; ++xx) {
                for (int yy = 0; yy < patch_size; ++yy) {
                    if (padded.at<float>(y + yy, x + xx) == target) {
                        t = {y + yy, x + xx};
                        found = true;
                        break;
                    }
                }
            }
            targets[static_cast<std::size_t>(index)] = t;
        }
    });

    for (int y = 0; y < source.rows; ++y) {
        for (int x = 0; x < source.cols; ++x) {
            float current_min = padded.at<float>(y, x);
            for (int yy = 0; yy < patch_size; ++yy)
                for (int xx = 0; xx < patch_size; ++xx)
                    current_min = std::min(current_min, padded.at<float>(y + yy, x + xx));

            const float refined = selected.at<uchar>(y, x) ? 0.0F : minima.at<float>(y, x);
            if (current_min != refined) {
                const Target t = targets[static_cast<std::size_t>(y) * source.cols + x];
                padded.at<float>(t.y, t.x) = refined;
            }
        }
    }

    cv::Mat out = padded(cv::Rect(radius, radius, source.cols, source.rows)).clone();
    if (radius) {
        source(cv::Rect(0, 0, source.cols, radius)).copyTo(out(cv::Rect(0, 0, source.cols, radius)));
        source(cv::Rect(0, source.rows - radius, source.cols, radius)).copyTo(out(cv::Rect(0, out.rows - radius, out.cols, radius)));
        source(cv::Rect(0, 0, radius, source.rows)).copyTo(out(cv::Rect(0, 0, radius, out.rows)));
        source(cv::Rect(source.cols - radius, 0, radius, source.rows)).copyTo(out(cv::Rect(out.cols - radius, 0, radius, out.rows)));
    }
    return out;
}

}  // namespace fast_deblur::internal
