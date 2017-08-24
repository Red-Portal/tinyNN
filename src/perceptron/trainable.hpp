#ifnef _TRAINABLE_HPP_
#define _TRAINABLE_HPP_

#include <blaze/math/StaticMatrix.h>

template<typename T, size_t Ydim>
using vector = blaze::StaticVector<float, Ydim>;

namespace perceptron
{
    template<typename T, size_t Xdim, size_t Ydim>
    class trainable
    {
    public:
        virtual void
        update_weight(vector<T, Ydim> const& error) = 0;
    }
}
