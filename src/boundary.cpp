#include "internal.hpp"

#include <fftw3.h>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace fast_deblur::internal {
namespace {

cv::Mat solveMinLaplacian(const cv::Mat& boundary_input) {
    CV_Assert(boundary_input.type() == CV_64F);
    cv::Mat boundary = boundary_input.clone();
    const int h = boundary.rows;
    const int w = boundary.cols;
    if (h <= 2 || w <= 2) return boundary;

    cv::Mat boundary_only = boundary.clone();
    boundary_only(cv::Rect(1, 1, w - 2, h - 2)).setTo(0.0);

    const int ih = h - 2;
    const int iw = w - 2;
    std::vector<double> rhs(static_cast<std::size_t>(ih) * iw, 0.0);
    for (int y = 1; y < h - 1; ++y) {
        for (int x = 1; x < w - 1; ++x) {
            const double fbp = -4.0 * boundary_only.at<double>(y, x)
                + boundary_only.at<double>(y, x + 1)
                + boundary_only.at<double>(y, x - 1)
                + boundary_only.at<double>(y - 1, x)
                + boundary_only.at<double>(y + 1, x);
            rhs[static_cast<std::size_t>(y - 1) * iw + (x - 1)] = -fbp;
        }
    }

    std::vector<double> transformed(rhs.size());
    fftw_plan forward = fftw_plan_r2r_2d(
        ih, iw, rhs.data(), transformed.data(), FFTW_RODFT00, FFTW_RODFT00, FFTW_ESTIMATE);
    if (!forward) throw std::runtime_error("failed to create FFTW DST plan");
    fftw_execute(forward);
    fftw_destroy_plan(forward);

    for (int y = 0; y < ih; ++y) {
        const double yy = static_cast<double>(y + 1);
        for (int x = 0; x < iw; ++x) {
            const double xx = static_cast<double>(x + 1);
            const double denom = 2.0 * std::cos(CV_PI * xx / (w - 1)) - 2.0
                               + 2.0 * std::cos(CV_PI * yy / (h - 1)) - 2.0;
            transformed[static_cast<std::size_t>(y) * iw + x] /= denom;
        }
    }

    std::vector<double> interior(transformed.size());
    fftw_plan inverse = fftw_plan_r2r_2d(
        ih, iw, transformed.data(), interior.data(), FFTW_RODFT00, FFTW_RODFT00, FFTW_ESTIMATE);
    if (!inverse) throw std::runtime_error("failed to create FFTW inverse DST plan");
    fftw_execute(inverse);
    fftw_destroy_plan(inverse);

    const double scale = 4.0 * static_cast<double>(ih + 1) * static_cast<double>(iw + 1);
    cv::Mat result = boundary_only;
    for (int y = 0; y < ih; ++y)
        for (int x = 0; x < iw; ++x)
            result.at<double>(y + 1, x + 1) = interior[static_cast<std::size_t>(y) * iw + x] / scale;
    return result;
}

cv::Mat wrapChannel(const cv::Mat& input, cv::Size target_shape) {
    CV_Assert(input.channels() == 1);
    cv::Mat image;
    input.convertTo(image, CV_64F);
    const int h = image.rows;
    const int w = image.cols;
    const int th = target_shape.height;
    const int tw = target_shape.width;
    const int h_extra = th - h;
    const int w_extra = tw - w;
    if (h_extra == 0 && w_extra == 0) return image;

    cv::Mat a = cv::Mat::zeros(h_extra + 2, w, CV_64F);
    image.row(h - 1).copyTo(a.row(0));
    image.row(0).copyTo(a.row(a.rows - 1));
    if (h_extra) {
        for (int y = 0; y < h_extra; ++y) {
            const double blend = h_extra == 1 ? 0.0 : static_cast<double>(y) / (h_extra - 1);
            a.at<double>(y + 1, 0) = (1.0 - blend) * a.at<double>(0, 0) + blend * a.at<double>(a.rows - 1, 0);
            a.at<double>(y + 1, w - 1) = (1.0 - blend) * a.at<double>(0, w - 1) + blend * a.at<double>(a.rows - 1, w - 1);
        }
    }
    a = solveMinLaplacian(a);

    cv::Mat b = cv::Mat::zeros(h, w_extra + 2, CV_64F);
    image.col(w - 1).copyTo(b.col(0));
    image.col(0).copyTo(b.col(b.cols - 1));
    if (w_extra) {
        for (int x = 0; x < w_extra; ++x) {
            const double blend = w_extra == 1 ? 0.0 : static_cast<double>(x) / (w_extra - 1);
            b.at<double>(0, x + 1) = (1.0 - blend) * b.at<double>(0, 0) + blend * b.at<double>(0, b.cols - 1);
            b.at<double>(h - 1, x + 1) = (1.0 - blend) * b.at<double>(h - 1, 0) + blend * b.at<double>(h - 1, b.cols - 1);
        }
    }
    b = solveMinLaplacian(b);

    cv::Mat c = cv::Mat::zeros(h_extra + 2, w_extra + 2, CV_64F);
    b.row(h - 1).copyTo(c.row(0));
    b.row(0).copyTo(c.row(c.rows - 1));
    a.col(w - 1).copyTo(c.col(0));
    a.col(0).copyTo(c.col(c.cols - 1));
    c = solveMinLaplacian(c);

    cv::Mat out = cv::Mat::zeros(th, tw, CV_64F);
    image.copyTo(out(cv::Rect(0, 0, w, h)));
    if (w_extra) b(cv::Rect(1, 0, w_extra, h)).copyTo(out(cv::Rect(w, 0, w_extra, h)));
    if (h_extra) a(cv::Rect(0, 0, w, h_extra)).copyTo(out(cv::Rect(0, h, w, h_extra)));
    if (h_extra && w_extra)
        c(cv::Rect(1, 1, w_extra, h_extra)).copyTo(out(cv::Rect(w, h, w_extra, h_extra)));
    return out;
}

}  // namespace

cv::Mat wrapBoundary(const cv::Mat& image, cv::Size target_shape) {
    if (target_shape.width < image.cols || target_shape.height < image.rows)
        throw std::invalid_argument("target shape must not be smaller than image");
    if (target_shape == image.size()) return image.clone();

    cv::Mat input = asFloat01(image);
    if (input.channels() == 1) {
        cv::Mat result = wrapChannel(input, target_shape);
        result.convertTo(result, CV_32F);
        return result;
    }
    std::vector<cv::Mat> planes;
    cv::split(input, planes);
    for (auto& plane : planes) {
        cv::Mat wrapped = wrapChannel(plane, target_shape);
        wrapped.convertTo(plane, CV_32F);
    }
    cv::Mat out;
    cv::merge(planes, out);
    return out;
}

}  // namespace fast_deblur::internal
