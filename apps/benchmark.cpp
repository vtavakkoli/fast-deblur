#include "fast_deblur/fast_deblur.hpp"

#include <nlohmann/json.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <cctype>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace {

struct Metrics {
    std::string image;
    std::string variant;
    double runtime_ms = 0.0;
    double reblur_rmse = 0.0;
    fast_deblur::ArtifactDiagnostics artifacts;
    std::string path;
};

const std::vector<std::pair<std::string, std::string>> kVariants = {
    {"matlab-parity", "MATLAB parity"},
    {"baseline", "Robust baseline"},
    {"trajectory", "Motion-Trajectory PSF"},
    {"annealed-pnp", "Annealed PnP"},
    {"extreme-channel", "Dual-Extreme"},
};

cv::Mat readRgb(const fs::path& path) {
    cv::Mat bgr = cv::imread(path.string(), cv::IMREAD_COLOR);
    if (bgr.empty()) throw std::runtime_error("cannot read " + path.string());
    cv::Mat rgb;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
    rgb.convertTo(rgb, CV_32FC3, 1.0 / 255.0);
    return rgb;
}

void writeRgb(const fs::path& path, const cv::Mat& rgb) {
    cv::Mat clipped;
    cv::max(rgb, 0.0, clipped);
    cv::min(clipped, 1.0, clipped);
    clipped.convertTo(clipped, CV_8UC3, 255.0);
    cv::Mat bgr;
    cv::cvtColor(clipped, bgr, cv::COLOR_RGB2BGR);
    if (!cv::imwrite(path.string(), bgr, {cv::IMWRITE_JPEG_QUALITY, 94})) {
        throw std::runtime_error("cannot write " + path.string());
    }
}

void writeKernel(const fs::path& path, const cv::Mat& kernel) {
    double peak = 0.0;
    cv::minMaxLoc(kernel, nullptr, &peak);
    cv::Mat vis = kernel.clone();
    if (peak > 0.0) vis /= static_cast<float>(peak);
    cv::resize(vis, vis, cv::Size(240, 240), 0.0, 0.0, cv::INTER_NEAREST);
    vis.convertTo(vis, CV_8U, 255.0);
    if (!cv::imwrite(path.string(), vis)) {
        throw std::runtime_error("cannot write " + path.string());
    }
}

fast_deblur::DeblurConfig configFromJson(const json& profile, bool preview) {
    fast_deblur::DeblurConfig config;
    config.kernel_size = profile.value("kernel_size", 25);
    config.gamma_correct = profile.value("gamma", 1.0F);
    config.lambda_dark = profile.value("lambda_dark", .004F);
    config.lambda_grad = profile.value("lambda_grad", .004F);
    config.lambda_tv = profile.value("lambda_tv", .003F);
    config.lambda_l0 = profile.value("lambda_l0", .0005F);
    config.weight_ring = profile.value("weight_ring", 1.0F);
    config.saturated = profile.value("saturated", false);
    if (preview) {
        config.xk_iter = 2;
        config.max_grad_steps = 5;
        config.max_dark_steps = 2;
        config.saturation_iterations = 12;
    }
    return config;
}

Metrics metric(
    const std::string& image,
    const std::string& variant,
    double runtime,
    const cv::Mat& source,
    const cv::Mat& output,
    const cv::Mat& kernel,
    const std::string& path) {
    Metrics result;
    result.image = image;
    result.variant = variant;
    result.runtime_ms = runtime;
    result.path = path;
    result.reblur_rmse = fast_deblur::rmse(source, fast_deblur::reblur(output, kernel));
    result.artifacts = fast_deblur::artifactDiagnostics(source, output);
    return result;
}

std::string safeStem(const std::string& name) {
    std::string stem = fs::path(name).stem().string();
    for (char& c : stem) {
        if (!(std::isalnum(static_cast<unsigned char>(c)) || c == '-' || c == '_')) c = '_';
    }
    return stem;
}

std::string escapeHtml(const std::string& text) {
    std::string out;
    for (char c : text) {
        if (c == '&') out += "&amp;";
        else if (c == '<') out += "&lt;";
        else if (c == '>') out += "&gt;";
        else if (c == '\"') out += "&quot;";
        else out += c;
    }
    return out;
}

std::string format(double value, int precision = 4) {
    std::ostringstream stream;
    stream << std::fixed << std::setprecision(precision) << value;
    return stream.str();
}

void writeReport(
    const fs::path& out,
    const std::vector<Metrics>& metrics,
    const std::vector<std::string>& images,
    const std::map<std::string, std::string>& errors,
    bool preview) {
    std::ofstream csv(out / "metrics.csv");
    csv << "image,variant,runtime_ms,reblur_rmse,edge_ratio,noise_ratio,highpass_ratio,clipping_growth,path\n";
    for (const auto& item : metrics) {
        csv << item.image << ',' << item.variant << ',' << item.runtime_ms << ',' << item.reblur_rmse << ','
            << item.artifacts.edge_ratio << ',' << item.artifacts.noise_ratio << ','
            << item.artifacts.highpass_ratio << ',' << item.artifacts.clipping_growth << ','
            << item.path << '\n';
    }

    std::map<std::string, std::pair<double, int>> runtime;
    std::map<std::string, std::pair<double, int>> residual;
    for (const auto& item : metrics) {
        runtime[item.variant].first += item.runtime_ms;
        runtime[item.variant].second++;
        residual[item.variant].first += item.reblur_rmse;
        residual[item.variant].second++;
    }

    std::ofstream html(out / "index.html");
    html << R"(<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>fast-deblur C++ benchmark</title><style>
body{font-family:Inter,system-ui,Segoe UI,Arial,sans-serif;margin:0;background:#fff;color:#17202a}main{max-width:1700px;margin:auto;padding:32px}h1{font-size:34px;margin-bottom:6px}.sub{color:#566573;max-width:1150px;line-height:1.55}.summary{display:flex;gap:14px;flex-wrap:wrap;margin:24px 0}.pill{border:1px solid #dfe6e9;border-radius:12px;padding:12px 16px;min-width:180px}.pill b{display:block;font-size:20px;margin-top:4px}.case{border-top:1px solid #e8ecef;padding:28px 0}.grid{display:grid;grid-template-columns:repeat(6,minmax(0,1fr));gap:12px}.panel{border:1px solid #e1e6ea;border-radius:12px;overflow:hidden;background:#fafbfc}.panel h3{font-size:13px;margin:0;padding:9px 11px;background:#f3f5f7}.panel img{display:block;width:100%;aspect-ratio:4/3;object-fit:contain;background:#fff}.metrics{font-size:12px;padding:9px 11px;line-height:1.5;color:#3c4a57}.kernels{display:flex;gap:12px;flex-wrap:wrap;margin-top:12px}.kernel img{width:160px;height:160px;image-rendering:pixelated;border:1px solid #ddd}.error{padding:12px;background:#fff2f0;border:1px solid #ffccc7;border-radius:8px}.note{font-size:13px;color:#65717c}@media(max-width:1250px){.grid{grid-template-columns:repeat(3,1fr)}}@media(max-width:760px){.grid{grid-template-columns:repeat(2,1fr)}}@media(max-width:520px){.grid{grid-template-columns:1fr}main{padding:18px}}
</style></head><body><main>)";

    html << "<h1>fast-deblur C++ benchmark</h1><p class='sub'>"
         << (preview ? "Preview benchmark with capped iteration counts." :
                       "Full-quality native-resolution benchmark with uncapped optimization loops.")
         << " MATLAB parity follows the translated legacy numerical path. Robust baseline is the current unconstrained blind estimator. "
            "Motion-Trajectory PSF reuses that robust result and adds a continuous movement-path constraint before guarded restoration. "
            "Annealed PnP and Dual-Extreme reuse the robust PSF. Saved upstream outputs are not inference targets.</p>";

    html << "<div class='summary'><div class='pill'>Images<b>" << images.size() << "</b></div>";
    for (const auto& [variant, values] : runtime) {
        html << "<div class='pill'>" << escapeHtml(variant) << " avg runtime<b>"
             << format(values.first / values.second, 1) << " ms</b><span class='note'>mean reblur RMSE "
             << format(residual[variant].first / residual[variant].second, 5) << "</span></div>";
    }
    html << "</div>";

    auto findMetric = [&](const std::string& image, const std::string& variant) -> const Metrics* {
        for (const auto& item : metrics) {
            if (item.image == image && item.variant == variant) return &item;
        }
        return nullptr;
    };

    for (const auto& name : images) {
        const std::string stem = safeStem(name);
        html << "<section class='case'><h2>" << escapeHtml(name) << "</h2>";
        if (auto error = errors.find(name); error != errors.end()) {
            html << "<div class='error'>" << escapeHtml(error->second) << "</div></section>";
            continue;
        }

        html << "<div class='grid'><div class='panel'><h3>Observed source</h3><img loading='lazy' src='images/"
             << stem << "_source.jpg'></div>";
        for (const auto& [variant, label] : kVariants) {
            const Metrics* item = findMetric(name, variant);
            html << "<div class='panel'><h3>" << label << "</h3><img loading='lazy' src='images/"
                 << stem << '_' << variant << ".jpg'>";
            if (item != nullptr) {
                html << "<div class='metrics'>runtime " << format(item->runtime_ms, 1)
                     << " ms<br>reblur RMSE " << format(item->reblur_rmse, 5)
                     << "<br>edge ×" << format(item->artifacts.edge_ratio, 2)
                     << " · noise ×" << format(item->artifacts.noise_ratio, 2)
                     << "<br>high-pass ×" << format(item->artifacts.highpass_ratio, 2)
                     << " · clip +" << format(item->artifacts.clipping_growth, 3)
                     << "<br><span class='note'>" << escapeHtml(item->path) << "</span></div>";
            }
            html << "</div>";
        }
        html << "</div><div class='kernels'>"
             << "<div class='kernel'><b>Parity PSF</b><br><img src='images/" << stem << "_parity_kernel.png'></div>"
             << "<div class='kernel'><b>Robust PSF</b><br><img src='images/" << stem << "_robust_kernel.png'></div>"
             << "<div class='kernel'><b>Motion-Trajectory PSF</b><br><img src='images/" << stem << "_trajectory_kernel.png'></div>"
             << "</div></section>";
    }

    html << "<p class='note'>Metrics are reference-free diagnostics. Reblur RMSE measures forward-model consistency; edge/noise/high-pass ratios and clipping growth flag common over-deconvolution artifacts. Baseline and MATLAB-parity runtimes include blind PSF estimation. Motion-Trajectory, Annealed PnP, and Dual-Extreme runtimes are incremental after the robust baseline has already been computed.</p></main></body></html>";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        fs::path data = "testdata/upstream/dataset/image";
        fs::path profiles = "benchmark_profiles.json";
        fs::path out = "report";
        int limit = 0;
        bool preview = false;

        for (int i = 1; i < argc; ++i) {
            const std::string argument = argv[i];
            auto value = [&]() {
                if (++i >= argc) throw std::invalid_argument("missing value for " + argument);
                return std::string(argv[i]);
            };
            if (argument == "--data-dir") data = value();
            else if (argument == "--profiles") profiles = value();
            else if (argument == "--out") out = value();
            else if (argument == "--limit") limit = std::stoi(value());
            else if (argument == "--preview") preview = true;
            else throw std::invalid_argument("unknown argument " + argument);
        }

        std::ifstream profile_file(profiles);
        if (!profile_file) throw std::runtime_error("cannot open profiles " + profiles.string());
        json profile_json;
        profile_file >> profile_json;

        fs::create_directories(out / "images");
        std::vector<Metrics> metrics;
        std::vector<std::string> images;
        std::map<std::string, std::string> errors;

        int processed = 0;
        for (auto iterator = profile_json.begin(); iterator != profile_json.end(); ++iterator) {
            if (limit > 0 && processed >= limit) break;
            const std::string name = iterator.key();
            images.push_back(name);
            ++processed;

            try {
                const cv::Mat source = readRgb(data / name);
                const std::string stem = safeStem(name);
                writeRgb(out / "images" / (stem + "_source.jpg"), source);

                auto config = configFromJson(iterator.value(), preview);
                auto parity_config = config;
                parity_config.robust_selection = false;
                parity_config.retry_gradient_only = false;
                parity_config.conservative_restoration = false;

                auto started = std::chrono::steady_clock::now();
                auto parity = fast_deblur::deblur(source, parity_config, fast_deblur::Method::Baseline);
                auto stopped = std::chrono::steady_clock::now();
                const double parity_ms = std::chrono::duration<double, std::milli>(stopped - started).count();
                writeRgb(out / "images" / (stem + "_matlab-parity.jpg"), parity.image);
                writeKernel(out / "images" / (stem + "_parity_kernel.png"), parity.kernel);
                metrics.push_back(metric(
                    name, "matlab-parity", parity_ms, source, parity.image, parity.kernel, parity.restoration_path));

                started = std::chrono::steady_clock::now();
                auto baseline = fast_deblur::deblur(source, config, fast_deblur::Method::Baseline);
                stopped = std::chrono::steady_clock::now();
                const double baseline_ms = std::chrono::duration<double, std::milli>(stopped - started).count();
                writeRgb(out / "images" / (stem + "_baseline.jpg"), baseline.image);
                writeKernel(out / "images" / (stem + "_robust_kernel.png"), baseline.kernel);
                metrics.push_back(metric(
                    name, "baseline", baseline_ms, source, baseline.image, baseline.kernel, baseline.restoration_path));

                started = std::chrono::steady_clock::now();
                auto trajectory = fast_deblur::applyMotionTrajectoryPrior(source, baseline, config);
                stopped = std::chrono::steady_clock::now();
                const double trajectory_ms = std::chrono::duration<double, std::milli>(stopped - started).count();
                writeRgb(out / "images" / (stem + "_trajectory.jpg"), trajectory.image);
                writeKernel(out / "images" / (stem + "_trajectory_kernel.png"), trajectory.kernel);
                metrics.push_back(metric(
                    name, "trajectory", trajectory_ms, source, trajectory.image, trajectory.kernel,
                    trajectory.restoration_path));

                started = std::chrono::steady_clock::now();
                cv::Mat pnp = fast_deblur::refine(
                    source, baseline.image, baseline.kernel, fast_deblur::Method::AnnealedPnP);
                stopped = std::chrono::steady_clock::now();
                const double pnp_ms = std::chrono::duration<double, std::milli>(stopped - started).count();
                writeRgb(out / "images" / (stem + "_annealed-pnp.jpg"), pnp);
                metrics.push_back(metric(
                    name, "annealed-pnp", pnp_ms, source, pnp, baseline.kernel, "reuse robust PSF"));

                started = std::chrono::steady_clock::now();
                cv::Mat extreme = fast_deblur::refine(
                    source, baseline.image, baseline.kernel, fast_deblur::Method::ExtremeChannel);
                stopped = std::chrono::steady_clock::now();
                const double extreme_ms = std::chrono::duration<double, std::milli>(stopped - started).count();
                writeRgb(out / "images" / (stem + "_extreme-channel.jpg"), extreme);
                metrics.push_back(metric(
                    name, "extreme-channel", extreme_ms, source, extreme, baseline.kernel,
                    "reuse robust PSF"));

                std::cout << "completed " << name << '\n';
            } catch (const std::exception& error) {
                errors[name] = error.what();
                std::cerr << "failed " << name << ": " << error.what() << '\n';
            }
        }

        writeReport(out, metrics, images, errors, preview);
        std::cout << "report: " << (out / "index.html") << '\n';
        return errors.empty() ? 0 : 3;
    } catch (const std::exception& error) {
        std::cerr << "benchmark: " << error.what() << '\n';
        return 1;
    }
}
