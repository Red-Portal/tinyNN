#ifndef _PERCEPTRON_HPP_
#define _PERCEPTRON_HPP_

#include <blaze/math/StaticMatrix.h>
#include <blaze/math/StaticVector.h>

#include "trainable.hpp"

namespace perceptron
{
    template<typename T, size_t Xdim, size_t Ydim>
    using matrix = blaze::StaticMatrix<T, Xdim, Ydim>;

    template<typename T, size_t Ydim>
    using vector = blaze::StaticVector<T, Ydim>;

    template<typename T, size_t Xdim, size_t Ydim>
    class perceptron : trainable<T, Xdim, Ydim>
    {
    private:
        vector<T, Ydim> weight;
        virtual void
        update_weight(vector<T, Ydim> const& correction) final;

    public:
        perceptron();

        virtual T
        operator()(vector<T, Xdim> const& input_x) const final;

        virtual vector<T, Ydim>
        operator()(matrix<T, Xdim, Ydim> const& input_x) const final;
    };
}

#endif
