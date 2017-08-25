#ifndef _PERCEPTRON_HPP_
#define _PERCEPTRON_HPP_

#include <cmath>

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
    class perceptron : trainable<T, InSize>
    {
    private:
        vector<T, InSize> weight;

        virtual void
        update_weight(vector<T, InSize> const& correction) final;

        inline T sigmoid(T) const noexcept;

    public:
        inline perceptron() = default;

        virtual T
        operator()(vector<T, InSize> const& input_x) const final;

        virtual vector_dyn<T>
        operator()(matrix_dyn<T> const& input_x) const final;
    };
}
#include "perceptron.tcc"

#endif
