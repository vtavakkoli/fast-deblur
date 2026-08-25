#include "internal.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <set>
#include <numeric>
#include <stdexcept>
#include <tuple>
#include <vector>

namespace fast_deblur::internal {

cv::Mat downsampleLevin(const cv::Mat& image, double ratio) {
    cv::Mat arr = asFloat01(image);
    if (std::abs(ratio - 1.0) < 1e-12) return arr;
    const double sigma = ratio / CV_PI;
    std::vector<double> kernel(101); double sum = 0.0;
    for (int i=-50;i<=50;++i){double grid=i*(2.0*CV_PI),value=std::exp(-0.5*grid*grid*sigma*sigma);kernel[static_cast<std::size_t>(i+50)]=value;sum+=value;}
    for(double& v:kernel)v/=sum;std::vector<double> cumulative(kernel.size());std::partial_sum(kernel.begin(),kernel.end(),cumulative.begin());std::vector<float> trimmed;
    for(std::size_t i=0;i<kernel.size();++i)if(std::min(cumulative[i],cumulative[kernel.size()-1-i])>0.05)trimmed.push_back(static_cast<float>(kernel[i]));
    cv::Mat k(1,static_cast<int>(trimmed.size()),CV_32F,trimmed.data()),filtered;cv::sepFilter2D(arr,filtered,CV_32F,k,k,cv::Point(-1,-1),0.0,cv::BORDER_CONSTANT);int radius=static_cast<int>(trimmed.size())/2;if(radius>0&&filtered.cols>2*radius&&filtered.rows>2*radius)filtered=filtered(cv::Rect(radius,radius,filtered.cols-2*radius,filtered.rows-2*radius)).clone();
    double step=1.0/ratio;int out_w=static_cast<int>(std::ceil(filtered.cols/step)),out_h=static_cast<int>(std::ceil(filtered.rows/step));cv::Mat map_x(out_h,out_w,CV_32F),map_y(out_h,out_w,CV_32F);for(int y=0;y<out_h;++y){float* mx=map_x.ptr<float>(y);float* my=map_y.ptr<float>(y);for(int x=0;x<out_w;++x){mx[x]=static_cast<float>(x*step);my[x]=static_cast<float>(y*step);}}cv::Mat out;cv::remap(filtered,out,map_x,map_y,cv::INTER_LINEAR,cv::BORDER_CONSTANT);return out;
}

std::pair<cv::Mat,cv::Mat> estimateBlurKernel(const cv::Mat& gray,const DeblurConfig& cfg){
    cfg.validate();cv::Mat y=asFloat01(gray);if(y.channels()!=1)throw std::invalid_argument("estimateBlurKernel expects grayscale input");if(cfg.prescale!=1.0F)cv::resize(y,y,cv::Size(),cfg.prescale,cfg.prescale,cv::INTER_CUBIC);if(cfg.gamma_correct!=1.0F){cv::max(y,0.0,y);cv::min(y,1.0,y);cv::pow(y,cfg.gamma_correct,y);}double ratio=std::sqrt(0.5);int max_iter=std::max(static_cast<int>(std::floor(std::log(5.0/cfg.kernel_size)/std::log(ratio))),0);std::vector<double> scales(max_iter+1);std::vector<int> sizes(max_iter+1);for(int i=0;i<=max_iter;++i){scales[i]=std::pow(ratio,i);int s=static_cast<int>(std::ceil(cfg.kernel_size*scales[i]));if(s%2==0)++s;sizes[i]=s;}
    std::optional<double> threshold;cv::Mat kernel,latent=y.clone();float ld=cfg.lambda_dark,lg=cfg.lambda_grad,peak_fraction=cfg.robust_selection?0.025F:0.05F,component_mass=cfg.robust_selection?0.025F:0.10F;
    for(int si=max_iter;si>=0;--si){int size=sizes[si];if(kernel.empty())kernel=initKernel(size);else kernel=resizeKernel(kernel,1.0/ratio,size);cv::Mat ys=downsampleLevin(y,scales[si]);cv::Mat padded=wrapBoundary(ys,fastShape(ys.size(),kernel.size())),visible=padded(cv::Rect(0,0,ys.cols,ys.rows));auto [bx,by]=validGradients(visible);if(!threshold){cv::Mat tx,ty;double t;std::tie(tx,ty,t)=thresholdGradients(ys,size,std::nullopt);threshold=t;}
        for(int iter=0;iter<cfg.xk_iter;++iter){if(ld!=0.0F){cv::Mat lp=l0DeblurDarkChannel(padded,kernel,ld,lg,cfg);latent=lp(cv::Rect(0,0,ys.cols,ys.rows)).clone();}else latent=l0Restoration(ys,kernel,lg,cfg);cv::Mat lx,ly;double t;std::tie(lx,ly,t)=thresholdGradients(latent,size,threshold);threshold=t;kernel=pruneKernel(estimatePsf(bx,by,lx,ly,2.0F,kernel.size(),peak_fraction),component_mass);if(ld!=0.0F)ld=std::max(ld/1.1F,1e-4F);lg=std::max(lg/1.1F,1e-4F);}kernel=adjustPsfCenter(kernel);
    }
    float kth=cfg.robust_selection?std::max(cfg.k_thresh,50.0F):cfg.k_thresh;double peak=0;cv::minMaxLoc(kernel,nullptr,&peak);if(kth>0&&peak>0)for(int yy=0;yy<kernel.rows;++yy){float* row=kernel.ptr<float>(yy);for(int xx=0;xx<kernel.cols;++xx)if(row[xx]<peak/kth)row[xx]=0;}cv::max(kernel,0.0,kernel);double total=cv::sum(kernel)[0];if(total<=0)kernel=initKernel(cfg.kernel_size);else kernel/=static_cast<float>(total);if(cfg.robust_selection)kernel=adjustPsfCenter(refinePsfStructure(kernel));cv::max(latent,0.0,latent);cv::min(latent,1.0,latent);return {kernel,latent};
}

namespace {
double severity(const ArtifactDiagnostics& d){return std::max(0.0,d.edge_ratio-1.80)+1.25*std::max(0.0,d.noise_ratio-1.70)+std::max(0.0,d.highpass_ratio-1.70)+25.0*d.clipping_growth;}
bool saturationInstability(const ArtifactDiagnostics& d){return d.edge_ratio>2.10&&d.noise_ratio>2.20;}
bool saturationSafe(const ArtifactDiagnostics& d){return d.edge_ratio<=1.80&&d.noise_ratio<=1.70&&d.highpass_ratio<=1.70&&d.clipping_growth<=0.020;}
bool preferCandidate(double cs,const ArtifactDiagnostics& cd,double ns,const ArtifactDiagnostics& nd,int ks){bool cr=rippleRisk(cd,ks),nr=rippleRisk(nd,ks);if(cr&&!nr)return ns<=cs*1.35;if(!cr&&nr)return false;return ns<cs;}
}  // namespace

std::tuple<cv::Mat,double,std::string> restoreAndScore(const cv::Mat& image,const cv::Mat& kernel,const DeblurConfig& cfg){
    if(cfg.saturated){int full=cfg.saturation_iterations;cv::Mat restored=whyteDeconvolution(image,kernel,full,cfg.threads);ArtifactDiagnostics fd;double score=restorationScore(image,restored,reblurImage(restored,kernel),&fd);if(!cfg.conservative_restoration||!saturationInstability(fd))return {restored,score,"whyte_"+std::to_string(full)};std::set<int> trials{std::max(5,static_cast<int>(std::lround(full*.30))),std::max(8,static_cast<int>(std::lround(full*.40))),std::max(10,static_cast<int>(std::lround(full*.60)))};cv::Mat best;ArtifactDiagnostics bd;double bs=INFINITY;int bi=-1;bool safe=false;for(int it:trials)if(it<full){cv::Mat c=whyteDeconvolution(image,kernel,it,cfg.threads);ArtifactDiagnostics d;double s=restorationScore(image,c,reblurImage(c,kernel),&d);if(saturationSafe(d)){if(!safe||it>bi){safe=true;best=c;bd=d;bs=s;bi=it;}}else if(!safe&&(best.empty()||severity(d)<severity(bd))){best=c;bd=d;bs=s;bi=it;}}if(!best.empty()&&severity(bd)<severity(fd)*.85)return {best,bs,"whyte_guarded_"+std::to_string(bi)};return {restored,score,"whyte_"+std::to_string(full)};}
    cv::Mat best=ringingArtifactsRemoval(image,kernel,cfg);ArtifactDiagnostics d;double score=restorationScore(image,best,reblurImage(best,kernel),&d);std::string name="configured";bool suspicious=score>.03||d.clipping_growth>.04||d.noise_ratio>1.8||rippleRisk(d,cfg.kernel_size);if(!cfg.conservative_restoration||!suspicious)return {best,score,name};DeblurConfig cc=cfg;cc.lambda_tv=std::max(cfg.lambda_tv*2.5F,1e-3F);cc.lambda_l0=std::max(cfg.lambda_l0*1.5F,7.5e-4F);cc.weight_ring=std::max(cfg.weight_ring,.65F);cv::Mat c=ringingArtifactsRemoval(image,kernel,cc);ArtifactDiagnostics cd;double cs=restorationScore(image,c,reblurImage(c,kernel),&cd);if(preferCandidate(score,d,cs,cd,cfg.kernel_size)){best=c;score=cs;d=cd;name="conservative";}bool still=score>.045||d.clipping_growth>.06||rippleRisk(d,cfg.kernel_size);if(still){DeblurConfig sc=cfg;sc.lambda_tv=std::max(cfg.lambda_tv*5.0F,2e-3F);sc.lambda_l0=std::max(cfg.lambda_l0*2.0F,1e-3F);sc.weight_ring=std::max(cfg.weight_ring,.85F);cv::Mat tv=ringingArtifactsRemoval(image,kernel,sc);ArtifactDiagnostics td;double ts=restorationScore(image,tv,reblurImage(tv,kernel),&td);if(preferCandidate(score,d,ts,td,cfg.kernel_size)){best=tv;score=ts;d=td;name="tv_safe";}}
    if(rippleRisk(d,cfg.kernel_size)){cv::Mat observed=asFloat01(image);for(float alpha:{.85F,.70F,.55F,.40F}){cv::Mat b=observed+alpha*(best-observed);cv::max(b,0.0,b);cv::min(b,1.0,b);ArtifactDiagnostics bd;double bs=restorationScore(image,b,reblurImage(b,kernel),&bd);if(!rippleRisk(bd,cfg.kernel_size)){best=b;score=bs;name="ripple_guard_"+std::to_string(alpha);break;}}}return {best,score,name};
}

}  // namespace fast_deblur::internal

namespace fast_deblur {
cv::Mat refine(const cv::Mat& observed,const cv::Mat& baseline,const cv::Mat& kernel,Method method){if(method==Method::Baseline)return internal::asFloat01(baseline);if(method==Method::AnnealedPnP)return internal::annealedPnpRefine(observed,baseline,kernel);if(method==Method::ExtremeChannel)return internal::extremeChannelRefine(observed,baseline,kernel);return internal::asFloat01(baseline);}
DeblurResult deblur(const cv::Mat& image,const DeblurConfig& config,Method method){DeblurConfig cfg=config;cfg.validate();if(cfg.threads>0)cv::setNumThreads(cfg.threads);cv::Mat arr=internal::asFloat01(image),gray=internal::matlabRgb2Gray(arr);auto [kernel,latent]=internal::estimateBlurKernel(gray,cfg);auto [result,score,path]=internal::restoreAndScore(arr,kernel,cfg);if(cfg.robust_selection&&cfg.retry_gradient_only&&!cfg.saturated&&cfg.lambda_dark!=0.0F&&internal::shouldRetryKernel(arr,result,kernel,cfg.kernel_size,score)){DeblurConfig retry=cfg;retry.lambda_dark=0.0F;retry.gamma_correct=1.0F;retry.retry_gradient_only=false;auto [rk,rl]=internal::estimateBlurKernel(gray,retry);auto [rr,rs,rpath]=internal::restoreAndScore(arr,rk,retry);auto pd=internal::diagnostics(arr,result),rd=internal::diagnostics(arr,rr);bool pr=internal::rippleRisk(pd,cfg.kernel_size),risk=internal::rippleRisk(rd,cfg.kernel_size);if((pr&&!risk&&rs<=score*1.35)||(!risk&&rs<score*.97)){kernel=rk;latent=rl;result=rr;score=rs;path="retry-gradient-only/"+rpath;}}if(method!=Method::Baseline){result=refine(arr,result,kernel,method);path+="+"+std::string(methodName(method));}return {result,kernel,latent,path};}
}  // namespace fast_deblur
