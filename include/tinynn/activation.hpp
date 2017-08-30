#ifndef _ACTIVATION_HPP_
#define _ACTIVATION_HPP_

#include <cmath>
#include <limits>

namespace tnn::activation_function
{
    inline double sigmoid(double x) noexcept
    { return std::exp(x) / (std::exp(x) + 1); }

    inline double identity(double x) noexcept
    { return x; }

    inline double binary_step(double x) noexcept
    { return x < 0 ? 0 : 1; }

    inline double rectified_linear(double x) noexcept
    { return x < 0 ? 0 : x; }

    template<double (*T)(double)>
    double derived(double x) noexcept;

    template<>
    double derived<sigmoid>(double x) noexcept
    { return sigmoid(x) * (1 - sigmoid(x)); }

    template<>
    double derived<identity>(double x) noexcept
    { (void)x;  return 1; }

    template<>
    double derived<binary_step>(double x) noexcept
    { return x == 0 ? std::numeric_limits<int32_t>::max() : 0; }

    template<>
    double derived<rectified_linear>(double x) noexcept
    { return x < 0 ? 0 : 1; }
}

#endif
