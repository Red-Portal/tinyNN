#ifndef _TRAINABLE_HPP_
#define _TRAINABLE_HPP_

#include <blaze/math/StaticVector.h>

namespace tnn
{
    template<typename T>
    using matrix_dyn = blaze::DynameMatrix<T>;

    template<typename T, size_t InSize>
    class trainable
    {
    public:
        virtual void
        update_weight(size_t layer_num, matrix_dyn<T> const& correction) = 0;

    public:
        virtual matrix_dyn<T>
        feed_layer(size_t layer_num, matrix_dyn<T> const& input) const = 0;
    };
}

#endif
