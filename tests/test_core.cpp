#include "fast_deblur/fast_deblur.hpp"
#include "internal.hpp"

#include <opencv2/imgproc.hpp>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {
void check(bool condition,const std::string& message){if(!condition)throw std::runtime_error(message);} 
bool finiteMat(const cv::Mat& m){cv::Mat flat=m.reshape(1);for(int y=0;y<flat.rows;++y){const float* row=flat.ptr<float>(y);for(int x=0;x<flat.cols;++x)if(!std::isfinite(row[x]))return false;}return true;}
void testPsfRoundTrip(){cv::Mat psf=cv::Mat::zeros(7,7,CV_32F);psf.at<float>(2,2)=.15F;psf.at<float>(3,3)=.55F;psf.at<float>(4,4)=.30F;cv::Mat r=fast_deblur::internal::otf2psf(fast_deblur::internal::psf2otf(psf,cv::Size(64,64)),psf.size());check(cv::norm(psf,r,cv::NORM_INF)<1e-4,"psf round trip failed");}
void testKernelInitialization(){cv::Mat k=fast_deblur::internal::initKernel(25);check(std::abs(cv::sum(k)[0]-1.0)<1e-6,"kernel not normalized");check(std::abs(k.at<float>(11,11)-.5F)<1e-6F&&std::abs(k.at<float>(11,12)-.5F)<1e-6F,"legacy taps wrong");}
void testChoFastShape(){cv::Size out=fast_deblur::internal::fastShape(cv::Size(257,193),cv::Size(25,25));check(out.width>=281&&out.height>=217,"fast FFT shape too small");}
void testBoundaryInvariant(){cv::Mat src(11,13,CV_32F);cv::randu(src,0.0F,1.0F);cv::Mat w=fast_deblur::internal::wrapBoundary(src,cv::Size(24,20));check(w.rows==20&&w.cols==24,"wrapped shape mismatch");check(cv::norm(src,w(cv::Rect(0,0,src.cols,src.rows)),cv::NORM_INF)<1e-5,"wrapping modified source");check(finiteMat(w),"non-finite boundary output");}
void testSequentialDarkProjectionBorder(){cv::Mat src(13,13,CV_32F);cv::randu(src,.1F,.9F);cv::Mat p=fast_deblur::internal::projectDarkChannel(src,.004F,.05F,5);check(p.size()==src.size(),"dark shape mismatch");check(cv::norm(src(cv::Rect(0,0,src.cols,2)),p(cv::Rect(0,0,src.cols,2)),cv::NORM_INF)<1e-6,"dark border changed");check(finiteMat(p),"non-finite dark output");}
void testSyntheticSmoke(){cv::Mat sharp=cv::Mat::zeros(72,96,CV_32FC3);cv::rectangle(sharp,cv::Rect(12,15,28,32),cv::Scalar(.85,.25,.15),cv::FILLED);cv::line(sharp,cv::Point(48,12),cv::Point(82,58),cv::Scalar(.2,.9,.4),4,cv::LINE_AA);cv::Mat known=cv::Mat::zeros(9,9,CV_32F);for(int i=2;i<=6;++i)known.at<float>(4,i)=.2F;cv::Mat observed=fast_deblur::reblur(sharp,known);fast_deblur::DeblurConfig cfg;cfg.kernel_size=9;cfg.dark_patch_size=7;cfg.xk_iter=1;cfg.max_grad_steps=2;cfg.max_dark_steps=1;cfg.robust_selection=false;cfg.retry_gradient_only=false;cfg.conservative_restoration=false;cfg.lambda_tv=.002F;cfg.lambda_l0=.001F;auto r=fast_deblur::deblur(observed,cfg,fast_deblur::Method::Baseline);check(r.image.size()==observed.size(),"deblur shape mismatch");check(r.kernel.rows==9&&r.kernel.cols==9,"kernel shape mismatch");check(std::abs(cv::sum(r.kernel)[0]-1.0)<5e-3,"estimated kernel not normalized");check(finiteMat(r.image)&&finiteMat(r.kernel),"non-finite deblur");cv::Mat pnp=fast_deblur::refine(observed,r.image,r.kernel,fast_deblur::Method::AnnealedPnP),ext=fast_deblur::refine(observed,r.image,r.kernel,fast_deblur::Method::ExtremeChannel);check(pnp.size()==observed.size()&&ext.size()==observed.size(),"refinement shape mismatch");check(finiteMat(pnp)&&finiteMat(ext),"non-finite refinement");}
}  // namespace
int main(){try{testPsfRoundTrip();testKernelInitialization();testChoFastShape();testBoundaryInvariant();testSequentialDarkProjectionBorder();testSyntheticSmoke();std::cout<<"All fast-deblur C++ tests passed.\n";return 0;}catch(const std::exception&e){std::cerr<<"TEST FAILURE: "<<e.what()<<'\n';return 1;}}
