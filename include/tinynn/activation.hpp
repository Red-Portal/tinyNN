#ifndef _ACTIVATION_HPP_
#define _ACTIVATION_HPP_

#include <cmath>

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
}

#endif
