#ifndef _MULTI_LAYER_PERCEPTRON_
#define _MULTI_LAYER_PERCEPTRON_

#include <vector>
#include <functional>

#include <blaze/math/DynamicMatrix.h>
#include <blaze/math/StaticVector.h>
#include <blaze/math/DynamicVector.h>

#include "trainable.hpp"

namespace tnn
{
    template<typename T>
    using matrix_dyn = blaze::DynamicMatrix<T>;

    template<typename T, size_t Xdim>
    using vector = blaze::StaticVector<T, Xdim>;

    template<typename T>
    using vector_dyn = blaze::DynamicVector<T>;

    template<typename T, size_t InSize>
    class multi_layer_perceptron
        : public trainable<T, InSize>
    {
    private:
        std::vector<matrix_dyn<T>> _layers;
        std::function<double(double)> _activation_fun;

        virtual void
        update_weight(size_t layer_num,
                      matrix_dyn<T> const& correction) final;

        inline std::vector<matrix_dyn<T>>
        set_layers(std::vector<size_t> const& layer_setting);

        virtual matrix_dyn<T>
        feed_layer(size_t layer_num,
                   matrix_dyn<T> const& input) const final;

        virtual matrix_dyn<T>
        fast_back_propagation(size_t layer_num,
                              matrix_dyn<T> const& delta) const final;

    public:
        inline multi_layer_perceptron(
            std::function<double(double)> const& activation_func,
            std::vector<size_t> const& layer_setting);

        vector_dyn<T>
        operator()(vector<T, InSize> const& x) const;

        matrix_dyn<T>
        operator()(matrix_dyn<T> const& x) const;
    };
}

#include "multi_layer_perceptron.tpp"

#endif
