#include <algorithm>

#include "perceptron.hpp"

namespace tnn 
{
    template<typename T, size_t InSize>
    void
    perceptron<T, InSize>::
    update_weight(vector<T, InSize> const& correction)
    {
        weight += correction;
    }

    template<typename T, size_t InSize>
    vector_dyn<T>
    perceptron<T, InSize>::
    operator()(matrix_dyn<T> const& input_x) const
    {
        vector_dyn<T> prod = input_x * weight;

        std::transform(prod.begin(), prod.end(),
                       prod.begin(),
                       std::mem_fun(&perceptron<T, InSize>::sigmoid));

        return prod;
    }

    template<typename T, size_t InSize>
    T perceptron<T, InSize>::
    sigmoid(T in) const noexcept
    {
        auto in_d = static_cast<double>(in);
        auto result = std::exp(in_d) / (std::exp(in_d) + 1);

        return static_cast<T>(result);
    }

    template<typename T, size_t InSize>
    T
    perceptron<T, InSize>::
    operator()(vector<T, InSize> const& input_x) const
    {
        return sigmoid(blaze::dot(input_x, weight));
    }
}
