#ifndef _PERCEPTRON_HPP_
#define _PERCEPTRON_HPP_

#include <cmath>
#include <functional>

#include <blaze/math/StaticMatrix.h>
#include <blaze/math/DynamicMatrix.h>
#include <blaze/math/StaticVector.h>
#include <blaze/math/DynamicVector.h>
#include <blaze/math/blas/gemv.h>

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
    class perceptron : public trainable<T, InSize>
    {
    private:
        vector<T, InSize> _weight;
        std::function<double(double)> _activation_func;

        virtual void
        update_weight(vector<T, InSize> const& correction) final;

        inline static T sigmoid(T) noexcept;

    public:
        inline explicit perceptron(
            std::function<double(double)> const& activation_func);

        virtual T
        operator()(vector<T, InSize> const& input_x) const final;

        virtual vector_dyn<T>
        operator()(matrix_dyn<T> const& input_x) const final;
    };
}
#include "perceptron.tcc"

#endif
