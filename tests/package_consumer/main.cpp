#include <fast_deblur/fast_deblur.hpp>

int main() {
    fast_deblur::DeblurConfig config;
    config.kernel_size = 25;
    config.validate();
    return fast_deblur::methodName(fast_deblur::Method::Baseline) == nullptr ? 1 : 0;
}
