#include "internal.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <tuple>
#include <vector>

namespace fast_deblur::internal {

std::pair<cv::Mat, cv::Mat> validGradients(const cv::Mat& image) {
    CV_Assert(image.type() == CV_32F);
    cv::Mat gx(image.rows - 1, image.cols - 1, CV_32F);
    cv::Mat gy(image.rows - 1, image.cols - 1, CV_32F);
    for (int y = 0; y < gx.rows; ++y) {
        const float* row = image.ptr<float>(y);
        const float* next = image.ptr<float>(y + 1);
        float* dx = gx.ptr<float>(y);
        float* dy = gy.ptr<float>(y);
        for (int x = 0; x < gx.cols; ++x) {
            dx[x] = row[x] - row[x + 1];
            dy[x] = row[x] - next[x];
        }
    }
    return {gx, gy};
}

std::tuple<cv::Mat, cv::Mat, double> thresholdGradients(
    const cv::Mat& latent, int psf_size, std::optional<double> threshold) {
    auto [px, py] = validGradients(latent);
    cv::Mat pm = px.mul(px) + py.mul(py);
    const bool first = !threshold.has_value();

    if (first) {
        constexpr double step = 0.00006;
        const int nsteps = static_cast<int>(std::floor(2.0 / step + 1e-12)) + 1;
        std::array<std::vector<long long>, 4> counts;
        for (auto& c : counts) c.assign(nsteps, 0);

        for (int y = 0; y < pm.rows; ++y) {
            const float* mx = px.ptr<float>(y);
            const float* my = py.ptr<float>(y);
            const float* mag = pm.ptr<float>(y);
            for (int x = 0; x < pm.cols; ++x) {
                double ratio;
                if (mx[x] == 0.0F) ratio = my[x] >= 0.0F ? INFINITY : -INFINITY;
                else ratio = static_cast<double>(my[x]) / static_cast<double>(mx[x]);
                const double pd = std::atan(ratio);
                int bin = -1;
                if (pd >= 0.0 && pd < CV_PI / 4.0) bin = 0;
                else if (pd >= CV_PI / 4.0 && pd < CV_PI / 2.0) bin = 1;
                else if (pd >= -CV_PI / 4.0 && pd < 0.0) bin = 2;
                else if (pd >= -CV_PI / 2.0 && pd < -CV_PI / 4.0) bin = 3;
                if (bin < 0) continue;
                const double v = mag[x];
                if (v < 0.0 || v > 2.0 + 1e-12) continue;
                int idx = static_cast<int>(std::floor(v / step));
                idx = std::clamp(idx, 0, nsteps - 1);
                counts[bin][idx]++;
            }
        }

        std::array<std::vector<long long>, 4> tails;
        for (int b = 0; b < 4; ++b) {
            tails[b].assign(nsteps, 0);
            long long cumulative = 0;
            for (int i = nsteps - 1, t = 0; i >= 0; --i, ++t) {
                cumulative += counts[b][i];
                tails[b][t] = cumulative;
            }
        }
        const long long required = std::max(psf_size * 20, 10);
        double selected = 0.0;
        for (int t = 0; t < nsteps; ++t) {
            long long minimum = tails[0][t];
            for (int b = 1; b < 4; ++b) minimum = std::min(minimum, tails[b][t]);
            if (minimum >= required) {
                selected = (nsteps - 1 - t) * step;
                break;
            }
        }
        threshold = selected;
    }

    double th = *threshold;
    auto allMasked = [&](double t) {
        for (int y = 0; y < pm.rows; ++y) {
            const float* row = pm.ptr<float>(y);
            for (int x = 0; x < pm.cols; ++x) if (row[x] >= t) return false;
        }
        return true;
    };
    double max_pm = 0.0;
    cv::minMaxLoc(pm, nullptr, &max_pm);
    while (allMasked(th) && max_pm > 0.0) th *= 0.81;

    for (int y = 0; y < pm.rows; ++y) {
        const float* mag = pm.ptr<float>(y);
        float* dx = px.ptr<float>(y);
        float* dy = py.ptr<float>(y);
        for (int x = 0; x < pm.cols; ++x) {
            if (mag[x] < th) dx[x] = dy[x] = 0.0F;
        }
    }
    if (!first) th /= 1.1;
    return {px, py, th};
}

namespace {

cv::Mat applyKernelNormal(const cv::Mat& x, const cv::Mat& spectrum, cv::Size image_shape, float weight) {
    cv::Mat xf = psf2otf(x, image_shape);
    cv::Mat product = complexTimesReal(xf, spectrum);
    cv::Mat out = otf2psf(product, x.size());
    out += weight * x;
    return out;
}

double dot(const cv::Mat& a, const cv::Mat& b) { return a.dot(b); }

cv::Mat normalizeNonnegative(const cv::Mat& input) {
    cv::Mat k;
    cv::max(input, 0.0, k);
    const double sum = cv::sum(k)[0];
    if (sum > 0) k /= static_cast<float>(sum);
    return k;
}

}  // namespace

cv::Mat estimatePsf(const cv::Mat& blurred_x, const cv::Mat& blurred_y,
                    const cv::Mat& latent_x, const cv::Mat& latent_y,
                    float weight, cv::Size psf_shape, float peak_fraction,
                    int max_iter, double tol) {
    if (peak_fraction < 0.0F || peak_fraction >= 1.0F) throw std::invalid_argument("peak_fraction must be in [0,1)");
    cv::Mat lxf = fftReal(latent_x), lyf = fftReal(latent_y);
    cv::Mat bxf = fftReal(blurred_x), byf = fftReal(blurred_y);
    cv::Mat term_x, term_y;
    cv::mulSpectrums(bxf, lxf, term_x, 0, true);
    cv::mulSpectrums(byf, lyf, term_y, 0, true);
    cv::Mat b = otf2psf(term_x + term_y, psf_shape);
    cv::Mat spectrum = abs2(lxf) + abs2(lyf);

    cv::Mat x(psf_shape, CV_32F, cv::Scalar(1.0F / static_cast<float>(psf_shape.area())));
    cv::Mat r = b - applyKernelNormal(x, spectrum, blurred_x.size(), weight);
    cv::Mat p = r.clone();
    double rsold = dot(r, r);
    for (int iter = 0; iter < max_iter; ++iter) {
        cv::Mat ap = applyKernelNormal(p, spectrum, blurred_x.size(), weight);
        const double denom = dot(p, ap);
        if (std::abs(denom) < 1e-20) break;
        const double alpha = rsold / denom;
        x += static_cast<float>(alpha) * p;
        r -= static_cast<float>(alpha) * ap;
        const double rsnew = dot(r, r);
        if (std::sqrt(rsnew) < tol) break;
        p = r + static_cast<float>(rsnew / std::max(rsold, 1e-30)) * p;
        rsold = rsnew;
    }

    double peak = 0.0;
    cv::minMaxLoc(x, nullptr, &peak);
    if (peak > 0 && peak_fraction > 0) {
        for (int y = 0; y < x.rows; ++y) {
            float* row = x.ptr<float>(y);
            for (int xx = 0; xx < x.cols; ++xx) if (row[xx] < peak * peak_fraction) row[xx] = 0.0F;
        }
    }
    const double total = cv::sum(x)[0];
    if (std::abs(total) <= 1e-12) {
        x.setTo(0.0F);
        x.at<float>(psf_shape.height / 2, psf_shape.width / 2) = 1.0F;
    } else x /= static_cast<float>(total);
    return x;
}

cv::Mat pruneKernel(const cv::Mat& kernel, float min_component_mass) {
    cv::Mat k = normalizeNonnegative(kernel);
    cv::Mat mask;
    cv::compare(k, 0.0, mask, cv::CMP_GT);
    cv::Mat labels;
    const int count = cv::connectedComponents(mask, labels, 8, CV_32S);
    for (int label = 1; label < count; ++label) {
        double mass = 0.0;
        for (int y = 0; y < k.rows; ++y) {
            const int* l = labels.ptr<int>(y); const float* row = k.ptr<float>(y);
            for (int x = 0; x < k.cols; ++x) if (l[x] == label) mass += row[x];
        }
        if (mass < min_component_mass) {
            for (int y = 0; y < k.rows; ++y) {
                const int* l = labels.ptr<int>(y); float* row = k.ptr<float>(y);
                for (int x = 0; x < k.cols; ++x) if (l[x] == label) row[x] = 0.0F;
            }
        }
    }
    return normalizeNonnegative(k);
}

cv::Mat adjustPsfCenter(const cv::Mat& kernel) {
    cv::Mat k = normalizeNonnegative(kernel);
    const double total = cv::sum(k)[0];
    if (total <= 0) return k;
    double cy = 0.0, cx = 0.0;
    for (int y = 0; y < k.rows; ++y) {
        const float* row = k.ptr<float>(y);
        for (int x = 0; x < k.cols; ++x) { cy += row[x] * y; cx += row[x] * x; }
    }
    cy /= total; cx /= total;
    const double ty = (k.rows - 1) / 2.0, tx = (k.cols - 1) / 2.0;
    const int sy = static_cast<int>(std::lround(ty - cy));
    const int sx = static_cast<int>(std::lround(tx - cx));
    cv::Mat out = cv::Mat::zeros(k.size(), CV_32F);
    const int src_y0 = std::max(0, -sy), src_y1 = std::min(k.rows, k.rows - sy);
    const int src_x0 = std::max(0, -sx), src_x1 = std::min(k.cols, k.cols - sx);
    if (src_y1 > src_y0 && src_x1 > src_x0) {
        k(cv::Rect(src_x0, src_y0, src_x1 - src_x0, src_y1 - src_y0)).copyTo(
            out(cv::Rect(src_x0 + sx, src_y0 + sy, src_x1 - src_x0, src_y1 - src_y0)));
    }
    return normalizeNonnegative(out);
}

cv::Mat initKernel(int size) {
    cv::Mat k = cv::Mat::zeros(size, size, CV_32F);
    const int row = size / 2 - 1, left = size / 2 - 1;
    k.at<float>(row, left) = 0.5F; k.at<float>(row, left + 1) = 0.5F;
    return k;
}

static cv::Mat fixKernelSize(cv::Mat k, int rows, int cols) {
    while (k.rows != rows || k.cols != cols) {
        if (k.rows > rows) {
            const double first = cv::sum(k.row(0))[0], last = cv::sum(k.row(k.rows - 1))[0];
            k = (first < last ? k.rowRange(1, k.rows) : k.rowRange(0, k.rows - 1)).clone();
        } else if (k.rows < rows) {
            const double first = cv::sum(k.row(0))[0], last = cv::sum(k.row(k.rows - 1))[0];
            cv::Mat padded = cv::Mat::zeros(k.rows + 1, k.cols, CV_32F);
            if (first < last) k.copyTo(padded.rowRange(0, k.rows)); else k.copyTo(padded.rowRange(1, k.rows + 1));
            k = padded;
        }
        if (k.cols > cols) {
            const double first = cv::sum(k.col(0))[0], last = cv::sum(k.col(k.cols - 1))[0];
            k = (first < last ? k.colRange(1, k.cols) : k.colRange(0, k.cols - 1)).clone();
        } else if (k.cols < cols) {
            const double first = cv::sum(k.col(0))[0], last = cv::sum(k.col(k.cols - 1))[0];
            cv::Mat padded = cv::Mat::zeros(k.rows, k.cols + 1, CV_32F);
            if (first < last) k.copyTo(padded.colRange(0, k.cols)); else k.copyTo(padded.colRange(1, k.cols + 1));
            k = padded;
        }
    }
    return k;
}

cv::Mat resizeKernel(const cv::Mat& kernel, double scale, int target_size) {
    const int new_h = std::max(1, static_cast<int>(std::ceil(kernel.rows * scale)));
    const int new_w = std::max(1, static_cast<int>(std::ceil(kernel.cols * scale)));
    cv::Mat k; cv::resize(kernel, k, cv::Size(new_w, new_h), 0.0, 0.0, cv::INTER_CUBIC);
    cv::max(k, 0.0, k); k = fixKernelSize(k, target_size, target_size);
    return normalizeNonnegative(k);
}

namespace {
struct AxisInfo { cv::Point2d center; cv::Point2d major; double anisotropy; };
AxisInfo principalAxis(const cv::Mat& k) {
    double peak = 0.0; cv::minMaxLoc(k, nullptr, &peak);
    std::vector<cv::Point2d> pts; std::vector<double> weights;
    for (int y = 0; y < k.rows; ++y) for (int x = 0; x < k.cols; ++x) { const float v=k.at<float>(y,x); if(v>=peak*0.12F){pts.emplace_back(x,y);weights.push_back(v);} }
    if (pts.size() < 3) { pts.clear(); weights.clear(); for (int y=0;y<k.rows;++y) for(int x=0;x<k.cols;++x){const float v=k.at<float>(y,x);if(v>0){pts.emplace_back(x,y);weights.push_back(v);}} }
    if (pts.size() < 2) return {{(k.cols-1)/2.0,(k.rows-1)/2.0},{1,0},1.0};
    double sw=std::accumulate(weights.begin(),weights.end(),0.0),cx=0,cy=0; for(std::size_t i=0;i<pts.size();++i){cx+=weights[i]*pts[i].x;cy+=weights[i]*pts[i].y;} cx/=sw;cy/=sw;
    cv::Mat cov=cv::Mat::zeros(2,2,CV_64F); for(std::size_t i=0;i<pts.size();++i){double dx=pts[i].x-cx,dy=pts[i].y-cy,w=weights[i];cov.at<double>(0,0)+=w*dx*dx;cov.at<double>(0,1)+=w*dx*dy;cov.at<double>(1,0)+=w*dx*dy;cov.at<double>(1,1)+=w*dy*dy;} cov/=sw;
    cv::Mat eval,evec;cv::eigen(cov,eval,evec);cv::Point2d major(evec.at<double>(0,0),evec.at<double>(0,1));double norm=std::hypot(major.x,major.y);major.x/=norm;major.y/=norm;return {{cx,cy},major,std::max(eval.at<double>(0),1e-9)/std::max(eval.at<double>(1),1e-9)};
}
cv::Mat weakLineMask(const cv::Mat& k) {
    double peak=0;cv::minMaxLoc(k,nullptr,&peak);cv::Mat support=k>=peak*0.015,strong=k>=peak*0.10,result=cv::Mat::zeros(k.size(),CV_8U);
    for(int y=0;y<k.rows;++y){std::vector<int> xs;double mass=0;int sc=0;for(int x=0;x<k.cols;++x)if(support.at<uchar>(y,x)){xs.push_back(x);mass+=k.at<float>(y,x);if(strong.at<uchar>(y,x))++sc;}if(xs.size()>=static_cast<std::size_t>(std::max(5,static_cast<int>(0.28*k.cols)))){int span=xs.back()-xs.front()+1;if(span>=static_cast<int>(0.38*k.cols)&&mass<=0.075&&sc<=std::max(2,static_cast<int>(0.04*k.cols)))for(int x:xs)result.at<uchar>(y,x)=255;}}
    for(int x=0;x<k.cols;++x){std::vector<int> ys;double mass=0;int sc=0;for(int y=0;y<k.rows;++y)if(support.at<uchar>(y,x)){ys.push_back(y);mass+=k.at<float>(y,x);if(strong.at<uchar>(y,x))++sc;}if(ys.size()>=static_cast<std::size_t>(std::max(5,static_cast<int>(0.28*k.rows)))){int span=ys.back()-ys.front()+1;if(span>=static_cast<int>(0.38*k.rows)&&mass<=0.075&&sc<=std::max(2,static_cast<int>(0.04*k.rows)))for(int y:ys)result.at<uchar>(y,x)=255;}}
    cv::Mat protected_core;cv::dilate(strong,protected_core,cv::Mat::ones(5,5,CV_8U));result.setTo(0,protected_core);return result;
}
}  // namespace

cv::Mat refinePsfStructure(const cv::Mat& kernel) {
    cv::Mat k=normalizeNonnegative(kernel);double peak=0;cv::minMaxLoc(k,nullptr,&peak);if(peak<=0)return k;
    cv::Mat support=k>=peak*0.015,labels;int n=cv::connectedComponents(support,labels,8,CV_32S);
    for(int label=1;label<n;++label){double mass=0;for(int y=0;y<k.rows;++y)for(int x=0;x<k.cols;++x)if(labels.at<int>(y,x)==label)mass+=k.at<float>(y,x);if(mass<0.012)for(int y=0;y<k.rows;++y)for(int x=0;x<k.cols;++x)if(labels.at<int>(y,x)==label)k.at<float>(y,x)=0;}
    cv::Mat weak=weakLineMask(k);k.setTo(0.0F,weak);k=normalizeNonnegative(k);AxisInfo axis=principalAxis(k);
    if(axis.anisotropy>=5.0){cv::minMaxLoc(k,nullptr,&peak);double band=std::max(2.0,0.085*std::max(k.rows,k.cols));cv::Point2d perp(-axis.major.y,axis.major.x);for(int y=0;y<k.rows;++y)for(int x=0;x<k.cols;++x){double dist=std::abs((x-axis.center.x)*perp.x+(y-axis.center.y)*perp.y);if(dist>band&&k.at<float>(y,x)<peak*0.055)k.at<float>(y,x)=0;}}
    return normalizeNonnegative(k);
}

}  // namespace fast_deblur::internal
