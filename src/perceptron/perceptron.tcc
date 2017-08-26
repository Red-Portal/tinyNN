#include <algorithm>
#include <random>

#include "perceptron.hpp"

namespace tnn 
{
    template<typename T, size_t InSize>
    perceptron<T, InSize>::
    perceptron()
        : weight()
    {
        auto reg_distr = std::uniform_real_distribution<double>(0, 1);
        auto seed_gen = std::random_device();
        auto rand_gen = std::mt19937(seed_gen());

        std::transform(weight.begin(), weight.end(),
                       weight.begin(),
                       [&rand_gen, &reg_distr](T& elem){
                           (void)elem;
                           return reg_distr(rand_gen);
                       });
    }

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
        blaze::DynamicVector<T> prod = input_x * weight;

        std::transform(prod.begin(), prod.end(),
                       prod.begin(),
                       &perceptron<T, InSize>::sigmoid);

        return prod;
    }

    template<typename T, size_t InSize>
    T perceptron<T, InSize>::
    sigmoid(T in) noexcept
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
