#include "internal.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace fast_deblur::internal {
namespace {
cv::Mat applyOtf(const cv::Mat& image,const cv::Mat& otf,bool conjugate){std::vector<cv::Mat> planes;if(image.channels()==1)planes.push_back(image);else cv::split(image,planes);for(auto& plane:planes){cv::Mat spec=fftReal(plane),prod;cv::mulSpectrums(spec,otf,prod,0,conjugate);plane=ifftReal(prod);}if(planes.size()==1)return planes[0];cv::Mat out;cv::merge(planes,out);return out;}
cv::Mat gaussianKernel2d(int size,double sigma){cv::Mat g=cv::getGaussianKernel(size,sigma,CV_32F);return g*g.t();}
}  // namespace

cv::Mat whyteDeconvolution(const cv::Mat& image,const cv::Mat& kernel,int iterations,int /*threads*/){
    if(iterations<1)throw std::invalid_argument("iterations must be >= 1");cv::Mat observed=asFloat01(image);std::vector<cv::Mat> op;if(observed.channels()==1)op.push_back(observed);else cv::split(observed,op);for(auto& p:op){cv::max(p,0.0,p);cv::pow(p,2.2,p);}cv::Mat linear;if(op.size()==1)linear=op[0];else cv::merge(op,linear);
    cv::Mat psf;cv::max(kernel,0.0,psf);double peak=0;cv::minMaxLoc(psf,nullptr,&peak);if(peak<=0)throw std::invalid_argument("kernel must contain positive mass");for(int y=0;y<psf.rows;++y){float* row=psf.ptr<float>(y);for(int x=0;x<psf.cols;++x)if(row[x]<peak/100.0)row[x]=0;}psf/=static_cast<float>(cv::sum(psf)[0]);
    int top=static_cast<int>(std::ceil((psf.rows-1)/2.0)),bottom=static_cast<int>(std::floor((psf.rows-1)/2.0)),left=static_cast<int>(std::ceil((psf.cols-1)/2.0)),right=static_cast<int>(std::floor((psf.cols-1)/2.0));cv::Mat padded;cv::copyMakeBorder(linear,padded,top,bottom,left,right,cv::BORDER_REPLICATE);cv::Mat kfft=psf2otf(psf,padded.size());cv::Mat support;cv::compare(psf,0.0,support,cv::CMP_NE);support.convertTo(support,CV_32F,1.0/255.0);cv::Mat sfft=psf2otf(support,padded.size());
    cv::Mat mask=cv::Mat::zeros(padded.size(),padded.type());cv::Rect valid(left,top,observed.cols,observed.rows);mask(valid).setTo(cv::Scalar::all(1.0));cv::Mat estimate=padded.clone();cv::Mat disk=cv::Mat::zeros(7,7,CV_8U);for(int y=0;y<7;++y)for(int x=0;x<7;++x)if((x-3)*(x-3)+(y-3)*(y-3)<=9)disk.at<uchar>(y,x)=1;cv::Mat smooth=gaussianKernel2d(21,3.0);int channels=estimate.channels();
    for(int iter=0;iter<iterations;++iter){cv::Mat pred=applyOtf(estimate,kfft,false);cv::max(pred,0.0,pred);cv::Mat sat(pred.size(),pred.type()),grad(pred.size(),pred.type());
        if(channels==1){for(int y=0;y<pred.rows;++y){const float* s=pred.ptr<float>(y);float* a=sat.ptr<float>(y);float* g=grad.ptr<float>(y);for(int x=0;x<pred.cols;++x){double z=50.0*(s[x]-1.0),sp=std::max(z,0.0)+std::log1p(std::exp(-std::abs(z)));a[x]=static_cast<float>(s[x]-sp/50.0);g[x]=static_cast<float>(1.0/(1.0+std::exp(std::clamp(z,-700.0,700.0))));}}}else{for(int y=0;y<pred.rows;++y){const cv::Vec3f* s=pred.ptr<cv::Vec3f>(y);cv::Vec3f* a=sat.ptr<cv::Vec3f>(y);cv::Vec3f* g=grad.ptr<cv::Vec3f>(y);for(int x=0;x<pred.cols;++x)for(int c=0;c<3;++c){double z=50.0*(s[x][c]-1.0),sp=std::max(z,0.0)+std::log1p(std::exp(-std::abs(z)));a[x][c]=static_cast<float>(s[x][c]-sp/50.0);g[x][c]=static_cast<float>(1.0/(1.0+std::exp(std::clamp(z,-700.0,700.0))));}}}
        cv::Mat ratio;cv::divide(padded,sat+1e-12F,ratio);cv::Mat err=(ratio-cv::Scalar::all(1.0)).mul(mask).mul(grad);std::vector<cv::Mat> ep,hp;if(channels==1)ep.push_back(estimate);else cv::split(estimate,ep);hp.resize(ep.size());for(std::size_t c=0;c<ep.size();++c){cv::Mat hard=ep[c]>=0.9F;cv::dilate(hard,hard,disk,cv::Point(-1,-1),1,cv::BORDER_CONSTANT,0);hard.convertTo(hp[c],CV_32F,1.0/255.0);}cv::Mat hard;if(hp.size()==1)hard=hp[0];else cv::merge(hp,hard);cv::Mat influence=applyOtf(hard,sfft,false);cv::min(influence,1.0,influence);cv::Mat valid_mask=1.0F-influence;
        cv::Mat uu=applyOtf(err.mul(valid_mask),kfft,true)+cv::Scalar::all(1.0),us=applyOtf(err,kfft,true)+cv::Scalar::all(1.0);std::vector<cv::Mat> wp(hp.size());for(std::size_t c=0;c<hp.size();++c)cv::filter2D(hp[c],wp[c],-1,smooth,cv::Point(-1,-1),0.0,cv::BORDER_CONSTANT);cv::Mat weights;if(wp.size()==1)weights=wp[0];else cv::merge(wp,weights);cv::Mat update=uu+(us-uu).mul(weights);cv::max(update,0.0,update);estimate=estimate.mul(update);
    }
    cv::Mat result=estimate(valid).clone();std::vector<cv::Mat> rp;if(result.channels()==1)rp.push_back(result);else cv::split(result,rp);for(auto& p:rp){cv::max(p,0.0,p);cv::pow(p,1.0/2.2,p);}if(rp.size()==1)return rp[0];cv::merge(rp,result);return result;
}

}  // namespace fast_deblur::internal
