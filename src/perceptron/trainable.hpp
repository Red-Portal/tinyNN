#ifndef _TRAINABLE_HPP_
#define _TRAINABLE_HPP_

#include <blaze/math/StaticMatrix.h>

namespace perceptron
{
    template<typename T, size_t Ydim>
    using vector = blaze::StaticVector<T, Ydim>;

    template<typename T, size_t Xdim, size_t Ydim>
    class trainable
    {
    public:
        virtual void
        update_weight(vector<T, Ydim> const& correction) = 0;

    public:
        virtual T
        operator()(vector<T, Xdim> const& input) const = 0;
    };
}

#endif
