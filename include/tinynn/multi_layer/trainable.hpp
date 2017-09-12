#ifndef _TRAINABLE_HPP_
#define _TRAINABLE_HPP_

#include <blaze/math/DynamicMatrix.h>
#include <blaze/math/DynamicVector.h>

namespace tnn
{
    template<typename T>
    using matrix_dyn = blaze::DynamicMatrix<T>;

    template<typename T, bool isRowVecT = false>
    using vector_dyn = blaze::DynamicVector<T>;

    template<typename T, size_t InSize>
    class trainable
    {
    public:
        virtual vector_dyn<T>
        eval_weight_delta(size_t layer_num,
                          vector_dyn<T> const& delta_o) const = 0;

        virtual void
        update_weight(size_t layer_num,
                      matrix_dyn<T> const& delta) = 0;

        virtual vector_dyn<T>
        feed_layer(size_t layer_num,
                   vector_dyn<T> const& input) const = 0;

        // virtual matrix_dyn<T>
        // fast_back_propagation(
        //     size_t layer_num,
        //     matrix_dyn<T> const& delta) const = 0;
    };
}

#endif
