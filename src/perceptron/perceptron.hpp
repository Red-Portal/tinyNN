#ifndef _PERCEPTRON_HPP_
#define _PERCEPTRON_HPP_

#include <blaze/math/StaticMatrix.h>
#include <trainable.hpp>

template<typename T, size_t Xdim, size_t Ydim>
using matrix = blaze::StaticMatrix<float, Xdim, Ydim>;

template<typename T, size_t Ydim>
using vector = blaze::StaticVector<float, Ydim>;

namespace perceptron
{
    template<typename T, size_t Xdim, size_t Ydim>
    class perceptron : trainable<Xdim, Ydim>
    {
    private:
        vector<T, Ydim> weight;

        virtual void
        update_weight(vector<T, Ydim> const& error) override final;

    public:
        perceptron();

        vector<T, Ydim>
        operator()(vector<T, Xdim, Ydim> const& input_x) const;
    }
}

#endif
