#include "multi_layer_perceptron.hpp"

namespace tnn
{
    template<typename T, size_t InSize>
    multi_layer_perceptron<T, InSize>::
    multi_layer_perceptron(std::function<double(double)> const& activation_func,
                           std::vector<size_t> const& layer_setting)
        : _layers(set_layers(layer_setting)),
          _activation_func(activation_func)
    {
        auto reg_distr = std::uniform_real_distribution<double>(0, 1);
        auto seed_gen = std::random_device();
        auto rand_gen = std::mt19937(seed_gen());

        for(auto& i : _layers)
        {
            i = map(i, [&rand_gen, &reg_distr](T elem)
                    {
                        (void)elem;
                        return reg_distr(rand_gen);
                    });
        }
    }

    template<typename T, size_t InSize>
    void
    multi_layer_perceptron<T, InSize>::
    update_weight(size_t layer_num, matrix_dyn<T> const& correction)
    {
        _layers[layer_num] += correction
    }

    template<typename T, size_t InSize>
    std::vector<matrix_dyn<T>>
    multi_layer_perceptron<T, InSize>::
    set_layers(std::vector<size_t> const& layer_setting) const
    {
        std::vector<marix_dyn> layers;
        layers.reserve(layer_setting.size());

        size_t previous_size = InSize;
        for(size_t i : layer_setting)
            layers.push_back(matrix_dyn(previous_size, i));

        return layers;
    }

    template<typename T, size_t InSize>
    matrix_dyn<T>
    multi_layer_perceptron<T, InSize>::
    feed_layer(size_t layer_num, matrix_dyn<T> const& input) const
    {
        return input * _layers[layer_num];
    }

    template<typename T, size_t InSize>
    vector_dyn<T>
    multi_layer_perceptron<T, InSize>::
    operator()(vector<T, InSize> const& x) const
    {
        for(auto const& layer : _layers)
            x = x * _layers;

        x = map(x, _activation_fun);
        return x;
    }

    template<typename T, size_t InSize>
    matrix_dyn<T>
    multi_layer_perceptron<T, InSize>::
    operator()(matrix_dyn<T, InSize> const& x) const
    {
        for(auto const& layer : _layers)
            x = x * _layers;

        x = map(x, _activation_fun);
        return x;
    }
}
