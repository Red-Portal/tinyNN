#ifndef _TRAINABLE_HPP_
#define _TRAINABLE_HPP_

#include <blaze/math/StaticVector.h>
#include <blaze/math/DynamicVector.h>

namespace tnn
{
    template<typename T, size_t Ydim>
    using vector = blaze::StaticVector<T, Ydim>;

    template<typename T, size_t InSize>
    class trainable
    {
    public:
        virtual void
        update_weight(vector<T, InSize> const& correction) = 0;

    public:
        virtual T
        operator()(vector<T, InSize> const& input) const = 0;
    };
}

#endif
