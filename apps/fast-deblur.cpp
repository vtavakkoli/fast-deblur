#include "fast_deblur/fast_deblur.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

void usage() {
    std::cout << R"(fast-deblur --input IMAGE --output IMAGE [options]

Methods:
  baseline          Robust blind PSF estimation and guarded restoration
  trajectory        Motion-Trajectory-Prior PSF projection + guarded restoration
  annealed-pnp      Robust baseline followed by Annealed PnP refinement
  extreme-channel   Robust baseline followed by Dual-Extreme refinement

Options:
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
)";
}

std::string requireValue(int& index, int argc, char** argv) {
    if (++index >= argc) {
        throw std::invalid_argument(std::string("missing value for ") + argv[index - 1]);
    }
    return argv[index];
}

cv::Mat readRgb(const std::string& path) {
    cv::Mat bgr = cv::imread(path, cv::IMREAD_COLOR);
    if (bgr.empty()) throw std::runtime_error("failed to read " + path);
    cv::Mat rgb;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
    rgb.convertTo(rgb, CV_32FC3, 1.0 / 255.0);
    return rgb;
}

void writeRgb(const std::string& path, const cv::Mat& rgb) {
    cv::Mat clipped;
    cv::max(rgb, 0.0, clipped);
    cv::min(clipped, 1.0, clipped);
    cv::Mat u8;
    clipped.convertTo(u8, CV_8UC3, 255.0);
    cv::Mat bgr;
    cv::cvtColor(u8, bgr, cv::COLOR_RGB2BGR);
    if (!cv::imwrite(path, bgr)) throw std::runtime_error("failed to write " + path);
}

void writeKernel(const std::string& path, const cv::Mat& kernel) {
    double peak = 0.0;
    cv::minMaxLoc(kernel, nullptr, &peak);
    cv::Mat vis = kernel.clone();
    if (peak > 0.0) vis /= static_cast<float>(peak);
    cv::Mat u8;
    vis.convertTo(u8, CV_8U, 255.0);
    if (!cv::imwrite(path, u8)) throw std::runtime_error("failed to write " + path);
}

}  // namespace

int main(int argc, char** argv) {
    try {
        std::string input;
        std::string output;
        std::string kernel_output;
        std::string method_name = "baseline";
        fast_deblur::Method method = fast_deblur::Method::Baseline;
        fast_deblur::DeblurConfig cfg;

        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--help" || arg == "-h") {
                usage();
                return 0;
            }
            if (arg == "--input") {
                input = requireValue(i, argc, argv);
            } else if (arg == "--output") {
                output = requireValue(i, argc, argv);
            } else if (arg == "--kernel-output") {
                kernel_output = requireValue(i, argc, argv);
            } else if (arg == "--method") {
                method_name = requireValue(i, argc, argv);
                if (method_name != "trajectory") method = fast_deblur::methodFromString(method_name);
            } else if (arg == "--kernel-size") {
                cfg.kernel_size = std::stoi(requireValue(i, argc, argv));
            } else if (arg == "--gamma") {
                cfg.gamma_correct = std::stof(requireValue(i, argc, argv));
            } else if (arg == "--lambda-dark") {
                cfg.lambda_dark = std::stof(requireValue(i, argc, argv));
            } else if (arg == "--lambda-grad") {
                cfg.lambda_grad = std::stof(requireValue(i, argc, argv));
            } else if (arg == "--lambda-tv") {
                cfg.lambda_tv = std::stof(requireValue(i, argc, argv));
            } else if (arg == "--lambda-l0") {
                cfg.lambda_l0 = std::stof(requireValue(i, argc, argv));
            } else if (arg == "--weight-ring") {
                cfg.weight_ring = std::stof(requireValue(i, argc, argv));
            } else if (arg == "--saturated") {
                cfg.saturated = true;
            } else if (arg == "--saturation-iterations") {
                cfg.saturation_iterations = std::stoi(requireValue(i, argc, argv));
            } else if (arg == "--threads") {
                cfg.threads = std::stoi(requireValue(i, argc, argv));
            } else if (arg == "--matlab-parity") {
                auto parity = fast_deblur::DeblurConfig::matlabParity();
                parity.kernel_size = cfg.kernel_size;
                parity.gamma_correct = cfg.gamma_correct;
                parity.lambda_dark = cfg.lambda_dark;
                parity.lambda_grad = cfg.lambda_grad;
                parity.lambda_tv = cfg.lambda_tv;
                parity.lambda_l0 = cfg.lambda_l0;
                parity.weight_ring = cfg.weight_ring;
                parity.saturated = cfg.saturated;
                parity.saturation_iterations = cfg.saturation_iterations;
                parity.threads = cfg.threads;
                cfg = parity;
            } else if (arg == "--fast") {
                cfg.xk_iter = 2;
                cfg.max_grad_steps = 5;
                cfg.max_dark_steps = 2;
                cfg.saturation_iterations = std::min(cfg.saturation_iterations, 12);
            } else {
                throw std::invalid_argument("unknown argument: " + arg);
            }
        }

        if (input.empty() || output.empty()) {
            usage();
            return 2;
        }

        const cv::Mat image = readRgb(input);
        fast_deblur::DeblurResult result = method_name == "trajectory"
            ? fast_deblur::deblurMotionTrajectory(image, cfg)
            : fast_deblur::deblur(image, cfg, method);

        writeRgb(output, result.image);
        if (!kernel_output.empty()) writeKernel(kernel_output, result.kernel);
        std::cout << "method=" << method_name
                  << " path=" << result.restoration_path
                  << " kernel=" << result.kernel.cols << 'x' << result.kernel.rows << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "fast-deblur: " << error.what() << '\n';
        return 1;
    }
}
