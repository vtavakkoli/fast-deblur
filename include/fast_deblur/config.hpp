#pragma once

#include <cstddef>
#include <stdexcept>
#include <string>

namespace fast_deblur {

enum class Method {
    Baseline,
    AnnealedPnP,
    ExtremeChannel,
};

inline Method methodFromString(const std::string& value) {
    if (value == "baseline") return Method::Baseline;
    if (value == "annealed-pnp") return Method::AnnealedPnP;
    if (value == "extreme-channel") return Method::ExtremeChannel;
    throw std::invalid_argument("unknown method: " + value);
}

inline const char* methodName(Method method) {
    switch (method) {
        case Method::Baseline: return "baseline";
        case Method::AnnealedPnP: return "annealed-pnp";
        case Method::ExtremeChannel: return "extreme-channel";
    }
    return "baseline";
}

struct DeblurConfig {
    int kernel_size = 25;
    float lambda_dark = 4e-3F;
    float lambda_grad = 4e-3F;
    float gamma_correct = 1.0F;
    int xk_iter = 5;
    float k_thresh = 20.0F;
    float prescale = 1.0F;

    float lambda_tv = 3e-3F;
    float lambda_l0 = 5e-4F;
    float weight_ring = 1.0F;
    bool saturated = false;
    int saturation_iterations = 50;

    int dark_patch_size = 35;
    float kappa = 2.0F;
    float beta_max_grad = 1e5F;
    float beta_max_pixel = 8.0F;

    bool robust_selection = true;
    bool retry_gradient_only = true;
    bool conservative_restoration = true;

    int max_grad_steps = -1;
    int max_dark_steps = -1;
    int threads = 0;

    void validate() const;
    static DeblurConfig matlabParity();
};

}  // namespace fast_deblur
