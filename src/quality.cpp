#include "internal.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace fast_deblur::internal {
namespace {

cv::Mat grayForQuality(const cv::Mat& image) {
    cv::Mat arr = asFloat01(image);
    if (arr.channels() == 1) return arr;
    cv::Mat gray; cv::cvtColor(arr, gray, cv::COLOR_RGB2GRAY); return gray;
}

double meanMagnitude(const cv::Mat& gx, const cv::Mat& gy) { cv::Mat mag; cv::magnitude(gx, gy, mag); return cv::mean(mag)[0]; }

double medianVector(std::vector<float> values) {
    if (values.empty()) return 0.0;
    const std::size_t n = values.size();
    std::nth_element(values.begin(), values.begin() + n / 2, values.end());
    double med = values[n / 2];
    if (n % 2 == 0) { auto max_it = std::max_element(values.begin(), values.begin() + n / 2); med = (med + *max_it) * 0.5; }
    return med;
}

double edgeEnergy(const cv::Mat& image) {
    cv::Mat g = grayForQuality(image), gx, gy; cv::Sobel(g, gx, CV_32F, 1, 0, 3); cv::Sobel(g, gy, CV_32F, 0, 1, 3); return meanMagnitude(gx, gy);
}

double noiseMad(const cv::Mat& image) {
    cv::Mat lap; cv::Laplacian(grayForQuality(image), lap, CV_32F);
    std::vector<float> values(lap.total()); std::copy(lap.ptr<float>(), lap.ptr<float>() + lap.total(), values.begin());
    const double median = medianVector(values); for (float& v : values) v = static_cast<float>(std::abs(v - median)); return medianVector(values);
}

double highpassRms(const cv::Mat& image) {
    cv::Mat arr = asFloat01(image), smooth; cv::GaussianBlur(arr, smooth, cv::Size(), 1.0, 1.0, cv::BORDER_REFLECT_101);
    cv::Mat sq = (arr - smooth).mul(arr - smooth); cv::Scalar means = cv::mean(sq); double mean = 0.0;
    for (int c = 0; c < arr.channels(); ++c) mean += means[c]; return std::sqrt(mean / arr.channels());
}

double clippingFraction(const cv::Mat& image, float margin = 1.0F / 255.0F) {
    cv::Mat arr = asFloat01(image); long long clipped = 0; const long long total = static_cast<long long>(arr.total()) * arr.channels();
    if (arr.channels() == 1) {
        for (int y=0;y<arr.rows;++y){const float* row=arr.ptr<float>(y);for(int x=0;x<arr.cols;++x)if(row[x]<=margin||row[x]>=1.0F-margin)++clipped;}
    } else {
        for (int y=0;y<arr.rows;++y){const cv::Vec3f* row=arr.ptr<cv::Vec3f>(y);for(int x=0;x<arr.cols;++x)for(int c=0;c<3;++c)if(row[x][c]<=margin||row[x][c]>=1.0F-margin)++clipped;}
    }
    return static_cast<double>(clipped) / std::max<long long>(total, 1);
}

int kernelComponentCount(const cv::Mat& kernel) {
    cv::Mat k; cv::max(kernel, 0.0, k); double peak=0; cv::minMaxLoc(k,nullptr,&peak); if(peak<=0)return 0;
    cv::Mat mask=k>=peak*0.05,labels; return std::max(0,cv::connectedComponents(mask,labels,8,CV_32S)-1);
}

}  // namespace

ArtifactDiagnostics diagnostics(const cv::Mat& observed, const cv::Mat& candidate) {
    ArtifactDiagnostics d;
    d.edge_ratio=edgeEnergy(candidate)/std::max(edgeEnergy(observed),0.02);
    d.noise_ratio=noiseMad(candidate)/std::max(noiseMad(observed),0.003);
    d.highpass_ratio=highpassRms(candidate)/std::max(highpassRms(observed),0.005);
    d.clipping_growth=std::max(0.0,clippingFraction(candidate)-clippingFraction(observed)); return d;
}

bool rippleRisk(const ArtifactDiagnostics& d, int kernel_size) {
    if(d.clipping_growth>=0.065)return true;
    if(kernel_size>=65&&d.clipping_growth>=0.025&&(d.edge_ratio>=2.50||d.highpass_ratio>=2.50))return true;
    if(kernel_size>=65)return d.edge_ratio>=2.60&&d.highpass_ratio>=3.50&&d.noise_ratio>=2.00;
    return false;
}

cv::Mat reblurImage(const cv::Mat& image, const cv::Mat& kernel) {
    cv::Mat arr=asFloat01(image); const int py=kernel.rows/2,px=kernel.cols/2; cv::Mat padded;
    cv::copyMakeBorder(arr,padded,py,py,px,px,cv::BORDER_REFLECT_101); cv::Mat otf=psf2otf(kernel,padded.size());
    std::vector<cv::Mat> planes; if(padded.channels()==1)planes.push_back(padded);else cv::split(padded,planes);
    for(auto& plane:planes){cv::Mat spec=fftReal(plane),blurred_spec;cv::mulSpectrums(spec,otf,blurred_spec,0,false);plane=ifftReal(blurred_spec);}
    cv::Mat blurred;if(planes.size()==1)blurred=planes[0];else cv::merge(planes,blurred);
    blurred=blurred(cv::Rect(px,py,arr.cols,arr.rows)).clone();cv::max(blurred,0.0,blurred);cv::min(blurred,1.0,blurred);return blurred;
}

double restorationScore(const cv::Mat& observed,const cv::Mat& candidate,const cv::Mat& reblurred,ArtifactDiagnostics* out_diag){
    const double r=fast_deblur::rmse(observed,reblurred);ArtifactDiagnostics d=diagnostics(observed,candidate);if(out_diag)*out_diag=d;double penalty=0;
    penalty+=0.006*std::pow(std::max(0.0,d.noise_ratio-1.45),2.0);penalty+=0.005*std::pow(std::max(0.0,d.edge_ratio-3.0),2.0);
    penalty+=0.004*std::pow(std::max(0.0,d.highpass_ratio-4.0),2.0);penalty+=0.18*d.clipping_growth;return r+penalty;
}

bool shouldRetryKernel(const cv::Mat& observed,const cv::Mat& restored,const cv::Mat& kernel,int kernel_size,double blind_score){
    ArtifactDiagnostics d=diagnostics(observed,restored);if(kernelComponentCount(kernel)>=3)return true;if(rippleRisk(d,kernel_size))return true;
    if(d.clipping_growth>0.07)return true;if(d.edge_ratio>3.0&&d.highpass_ratio>4.0)return true;if(kernel_size>=65&&d.edge_ratio<1.15&&blind_score>0.025)return true;
    if(kernel_size>=65&&blind_score>0.045)return true;return false;
}

}  // namespace fast_deblur::internal

namespace fast_deblur {
cv::Mat reblur(const cv::Mat& image,const cv::Mat& kernel){return internal::reblurImage(image,kernel);} 
ArtifactDiagnostics artifactDiagnostics(const cv::Mat& observed,const cv::Mat& candidate){return internal::diagnostics(observed,candidate);} 

double rmse(const cv::Mat& a,const cv::Mat& b){cv::Mat af=internal::asFloat01(a),bf=internal::asFloat01(b);if(af.size()!=bf.size()||af.channels()!=bf.channels())throw std::invalid_argument("rmse shape mismatch");cv::Mat diff=af-bf,sq=diff.mul(diff);cv::Scalar mean=cv::mean(sq);double value=0;for(int c=0;c<af.channels();++c)value+=mean[c];return std::sqrt(value/af.channels());}
double psnr(const cv::Mat& reference,const cv::Mat& candidate){const double e=rmse(reference,candidate);return e<=1e-12?INFINITY:20.0*std::log10(1.0/e);} 
double ssim(const cv::Mat& reference,const cv::Mat& candidate){cv::Mat a=internal::matlabRgb2Gray(internal::asFloat01(reference)),b=internal::matlabRgb2Gray(internal::asFloat01(candidate));cv::Scalar ma,sa,mb,sb;cv::meanStdDev(a,ma,sa);cv::meanStdDev(b,mb,sb);double mua=ma[0],mub=mb[0],vara=sa[0]*sa[0],varb=sb[0]*sb[0];cv::Mat ca=a-mua,cb=b-mub;double cov=cv::mean(ca.mul(cb))[0];constexpr double c1=.0001,c2=.0009;return((2*mua*mub+c1)*(2*cov+c2))/((mua*mua+mub*mub+c1)*(vara+varb+c2));}
}  // namespace fast_deblur
