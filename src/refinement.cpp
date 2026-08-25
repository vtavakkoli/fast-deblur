#include "internal.hpp"

#include <opencv2/imgproc.hpp>
#include <opencv2/photo.hpp>

#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>
#include <vector>

namespace fast_deblur::internal {
namespace {
cv::Mat clip01(cv::Mat value){cv::max(value,0.0,value);cv::min(value,1.0,value);return value;}
cv::Mat dataConsistency(const cv::Mat& observed,const cv::Mat& prior,const cv::Mat& kernel,float rho){
    if(rho<=0)throw std::invalid_argument("rho must be > 0");cv::Mat y=asFloat01(observed),z=asFloat01(prior);if(y.size()!=z.size()||y.channels()!=z.channels())throw std::invalid_argument("observed/prior shape mismatch");
    int py=kernel.rows/2,px=kernel.cols/2;cv::Mat yp,zp;cv::copyMakeBorder(y,yp,py,py,px,px,cv::BORDER_REFLECT_101);cv::copyMakeBorder(z,zp,py,py,px,px,cv::BORDER_REFLECT_101);
    cv::Mat otf=psf2otf(kernel,yp.size()),denominator=abs2(otf)+rho;std::vector<cv::Mat> ys,zs;if(y.channels()==1){ys.push_back(yp);zs.push_back(zp);}else{cv::split(yp,ys);cv::split(zp,zs);} 
    for(std::size_t c=0;c<ys.size();++c){cv::Mat yfft=fftReal(ys[c]),zfft=fftReal(zs[c]),fidelity;cv::mulSpectrums(yfft,otf,fidelity,0,true);ys[c]=ifftReal(complexDivideReal(fidelity+complexScale(zfft,rho),denominator));}
    cv::Mat estimate;if(ys.size()==1)estimate=ys[0];else cv::merge(ys,estimate);estimate=estimate(cv::Rect(px,py,y.cols,y.rows)).clone();return clip01(estimate);
}
cv::Mat nlmDenoiser(const cv::Mat& image,float sigma){cv::Mat arr=asFloat01(image),u8;arr.convertTo(u8,CV_MAKETYPE(CV_8U,arr.channels()),255.0);float h=std::clamp(sigma*255.0F*0.55F,2.0F,12.0F);cv::Mat out;if(u8.channels()==1)cv::fastNlMeansDenoising(u8,out,h,7,21);else{cv::Mat bgr,den;cv::cvtColor(u8,bgr,cv::COLOR_RGB2BGR);cv::fastNlMeansDenoisingColored(bgr,den,h,h,7,21);cv::cvtColor(den,out,cv::COLOR_BGR2RGB);}out.convertTo(out,CV_MAKETYPE(CV_32F,out.channels()),1.0/255.0);return out;}
cv::Mat artifactSafeBlend(const cv::Mat& observed,const cv::Mat& initial,const cv::Mat& candidate,const cv::Mat& kernel){
    cv::Mat y=asFloat01(observed),base=clip01(asFloat01(initial)),cand=clip01(asFloat01(candidate));int ks=std::max(kernel.rows,kernel.cols);auto bd=diagnostics(y,base),cd=diagnostics(y,cand);bool br=rippleRisk(bd,ks),cr=rippleRisk(cd,ks);if(cr)return base;if(br&&(cd.noise_ratio>=bd.noise_ratio*0.95||cd.highpass_ratio>=bd.highpass_ratio*0.95))return base;
    cv::Mat rb=reblurImage(base,kernel),rc=reblurImage(cand,kernel);double be=fast_deblur::rmse(rb,y),ce=fast_deblur::rmse(rc,y);if(!std::isfinite(ce)||ce>=be*0.997)return base;double bs=restorationScore(y,base,rb),cs=restorationScore(y,cand,rc);if(cs>=bs*0.995)return base;
    double gain=(be-ce)/std::max(be,1e-8),nr=cd.noise_ratio/std::max(bd.noise_ratio,5e-4),hr=cd.highpass_ratio/std::max(bd.highpass_ratio,2e-3),er=cd.edge_ratio/std::max(bd.edge_ratio,0.02),cg=std::max(0.0,cd.clipping_growth-bd.clipping_growth);double an=1.20+0.65*std::min(gain/0.30,1.0),ah=1.20+0.55*std::min(gain/0.30,1.0);if(nr>an*1.35||hr>ah*1.35||er>1.55||cg>0.025)return base;
    float alpha=static_cast<float>(std::clamp(std::clamp(gain/0.18,0.10,1.0)*std::min(1.0,an/std::max(nr,1e-8))*std::min(1.0,ah/std::max(hr,1e-8)),0.0,1.0));cv::Mat blended=clip01(base+alpha*(cand-base));if(rippleRisk(diagnostics(y,blended),ks))return base;if(restorationScore(y,blended,reblurImage(blended,kernel))>=bs)return base;return blended;
}
}  // namespace

cv::Mat annealedPnpRefine(const cv::Mat& observed,const cv::Mat& initial,const cv::Mat& kernel,int steps,float sigma_start,float sigma_end,int candidates,unsigned seed){
    if(steps<1||candidates<1)throw std::invalid_argument("steps/candidates must be >= 1");cv::Mat y=asFloat01(observed),base=clip01(asFloat01(initial)),x=base.clone();std::mt19937 rng(seed);
    for(int i=0;i<steps;++i){double t=steps==1?0.0:static_cast<double>(i)/(steps-1);float sigma=static_cast<float>(std::exp(std::log(sigma_start)*(1.0-t)+std::log(sigma_end)*t)),rho=0.05F+0.10F*static_cast<float>(t);double best_score=INFINITY;cv::Mat best=x;std::normal_distribution<float> dist(0.0F,sigma);
        for(int c=0;c<candidates;++c){cv::Mat noise(x.size(),x.type());if(x.channels()==1){for(int yy=0;yy<x.rows;++yy){float* row=noise.ptr<float>(yy);for(int xx=0;xx<x.cols;++xx)row[xx]=dist(rng);}}else{for(int yy=0;yy<x.rows;++yy){cv::Vec3f* row=noise.ptr<cv::Vec3f>(yy);for(int xx=0;xx<x.cols;++xx)for(int ch=0;ch<3;++ch)row[xx][ch]=dist(rng);}}cv::Mat candidate=dataConsistency(y,nlmDenoiser(clip01(x+noise),sigma),kernel,rho);double score=restorationScore(y,candidate,reblurImage(candidate,kernel));if(score<best_score){best_score=score;best=candidate;}}
        x=best;}
    return artifactSafeBlend(y,base,x,kernel);
}

cv::Mat extremeChannelRefine(const cv::Mat& observed,const cv::Mat& initial,const cv::Mat& kernel,int steps,int patch_size){
    if(steps<1)throw std::invalid_argument("steps must be >= 1");if(patch_size<3||patch_size%2==0)throw std::invalid_argument("patch_size must be odd and >= 3");cv::Mat y=asFloat01(observed),base=clip01(asFloat01(initial)),x=base.clone();if(x.channels()!=3)return x;cv::Mat footprint=cv::Mat::ones(patch_size,patch_size,CV_8U);
    for(int i=0;i<steps;++i){std::vector<cv::Mat> p;cv::split(x,p);cv::Mat minc,maxc;cv::min(p[0],p[1],minc);cv::min(minc,p[2],minc);cv::max(p[0],p[1],maxc);cv::max(maxc,p[2],maxc);cv::Mat dark,bright;cv::erode(minc,dark,footprint);cv::dilate(maxc,bright,footprint);cv::Mat dw=(0.10F-dark)/0.10F,bw=(bright-0.90F)/0.10F;cv::max(dw,0.0,dw);cv::min(dw,1.0,dw);cv::max(bw,0.0,bw);cv::min(bw,1.0,bw);cv::Mat extreme;cv::max(dw,bw,extreme);cv::Mat smooth;cv::bilateralFilter(x,smooth,0,0.06,2.0);cv::Mat detail=x-smooth;std::vector<cv::Mat> pp,dp;cv::split(x,pp);cv::split(detail,dp);for(int ch=0;ch<3;++ch){cv::Mat gain=0.04F+0.08F*extreme;pp[ch]=pp[ch]+gain.mul(dp[ch]);pp[ch]=pp[ch].mul(1.0F-0.010F*dw);pp[ch]=1.0F-(1.0F-pp[ch]).mul(1.0F-0.010F*bw);}cv::Mat prior;cv::merge(pp,prior);x=dataConsistency(y,clip01(prior),kernel,0.12F+0.06F*i);}
    return artifactSafeBlend(y,base,x,kernel);
}

}  // namespace fast_deblur::internal
