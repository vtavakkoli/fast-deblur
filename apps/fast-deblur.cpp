#include "fast_deblur/fast_deblur.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {
void usage(){std::cout<<R"(fast-deblur --input IMAGE --output IMAGE [options]
Options:
  --method baseline|annealed-pnp|extreme-channel
  --kernel-output PATH
  --kernel-size N
  --gamma X
  --lambda-dark X
  --lambda-grad X
  --lambda-tv X
  --lambda-l0 X
  --weight-ring X
  --saturated
  --saturation-iterations N
  --matlab-parity        Disable robust selection/guards for strict legacy parity
  --fast                 Preview caps: 2 blind alternations, 5 grad steps, 2 dark steps
  --threads N
)";}
std::string requireValue(int& i,int argc,char** argv){if(++i>=argc)throw std::invalid_argument(std::string("missing value for ")+argv[i-1]);return argv[i];}
cv::Mat readRgb(const std::string& path){cv::Mat bgr=cv::imread(path,cv::IMREAD_COLOR);if(bgr.empty())throw std::runtime_error("failed to read "+path);cv::Mat rgb;cv::cvtColor(bgr,rgb,cv::COLOR_BGR2RGB);rgb.convertTo(rgb,CV_32FC3,1.0/255.0);return rgb;}
void writeRgb(const std::string& path,const cv::Mat& rgb){cv::Mat clipped;cv::max(rgb,0.0,clipped);cv::min(clipped,1.0,clipped);cv::Mat u8;clipped.convertTo(u8,CV_8UC3,255.0);cv::Mat bgr;cv::cvtColor(u8,bgr,cv::COLOR_RGB2BGR);if(!cv::imwrite(path,bgr))throw std::runtime_error("failed to write "+path);}
void writeKernel(const std::string& path,const cv::Mat& kernel){double peak=0;cv::minMaxLoc(kernel,nullptr,&peak);cv::Mat vis=kernel.clone();if(peak>0)vis/=static_cast<float>(peak);cv::Mat u8;vis.convertTo(u8,CV_8U,255.0);if(!cv::imwrite(path,u8))throw std::runtime_error("failed to write "+path);}
}  // namespace

int main(int argc,char** argv){try{std::string input,output,kernel_output;fast_deblur::Method method=fast_deblur::Method::Baseline;fast_deblur::DeblurConfig cfg;for(int i=1;i<argc;++i){std::string arg=argv[i];if(arg=="--help"||arg=="-h"){usage();return 0;}else if(arg=="--input")input=requireValue(i,argc,argv);else if(arg=="--output")output=requireValue(i,argc,argv);else if(arg=="--kernel-output")kernel_output=requireValue(i,argc,argv);else if(arg=="--method")method=fast_deblur::methodFromString(requireValue(i,argc,argv));else if(arg=="--kernel-size")cfg.kernel_size=std::stoi(requireValue(i,argc,argv));else if(arg=="--gamma")cfg.gamma_correct=std::stof(requireValue(i,argc,argv));else if(arg=="--lambda-dark")cfg.lambda_dark=std::stof(requireValue(i,argc,argv));else if(arg=="--lambda-grad")cfg.lambda_grad=std::stof(requireValue(i,argc,argv));else if(arg=="--lambda-tv")cfg.lambda_tv=std::stof(requireValue(i,argc,argv));else if(arg=="--lambda-l0")cfg.lambda_l0=std::stof(requireValue(i,argc,argv));else if(arg=="--weight-ring")cfg.weight_ring=std::stof(requireValue(i,argc,argv));else if(arg=="--saturated")cfg.saturated=true;else if(arg=="--saturation-iterations")cfg.saturation_iterations=std::stoi(requireValue(i,argc,argv));else if(arg=="--threads")cfg.threads=std::stoi(requireValue(i,argc,argv));else if(arg=="--matlab-parity"){auto p=fast_deblur::DeblurConfig::matlabParity();p.kernel_size=cfg.kernel_size;p.gamma_correct=cfg.gamma_correct;p.lambda_dark=cfg.lambda_dark;p.lambda_grad=cfg.lambda_grad;p.lambda_tv=cfg.lambda_tv;p.lambda_l0=cfg.lambda_l0;p.weight_ring=cfg.weight_ring;p.saturated=cfg.saturated;p.saturation_iterations=cfg.saturation_iterations;p.threads=cfg.threads;cfg=p;}else if(arg=="--fast"){cfg.xk_iter=2;cfg.max_grad_steps=5;cfg.max_dark_steps=2;cfg.saturation_iterations=std::min(cfg.saturation_iterations,12);}else throw std::invalid_argument("unknown argument: "+arg);}if(input.empty()||output.empty()){usage();return 2;}auto result=fast_deblur::deblur(readRgb(input),cfg,method);writeRgb(output,result.image);if(!kernel_output.empty())writeKernel(kernel_output,result.kernel);std::cout<<"method="<<fast_deblur::methodName(method)<<" path="<<result.restoration_path<<" kernel="<<result.kernel.cols<<"x"<<result.kernel.rows<<"\n";return 0;}catch(const std::exception& e){std::cerr<<"fast-deblur: "<<e.what()<<"\n";return 1;}}
