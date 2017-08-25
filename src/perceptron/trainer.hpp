#ifndef _LEARNING_DEVICE_HPP_
#define _LEARNING_DEVICE_HPP_

#include <stdint.h>
#include <vector>
#include <memory>
#include <random>

#include <blaze/math/StaticMatrix.h>
#include <blaze/math/Submatrix.h>

#include "trainable.hpp"
#include "perceptron.hpp"

template<typename T, size_t Xdim, size_t Ydim>
using matrix = blaze::StaticMatrix<float, Xdim, Ydim>;


namespace perceptron
{
    template<typename T, size_t Xdim, size_t Ydim>
    class trainer
    {
        using matrix_view =
            typename blaze::Submatrix<matrix<float, Xdim, Ydim>>;
    private:
        double _eta;
        uint64_t _max_iterations;
        std::shared_ptr<perceptron<T, Xdim, Ydim>> _trainable;
        std::vector<T> _error;
        std::mt19937_64 _random_engine;
        std::uniform_int_distribution<size_t> _distr;

        inline matrix_view const& 
        pick_data_random(matrix<T, Xdim, Ydim> const& data_set) const;

        vector<T, Ydim> const&
        train_set(matrix_view const& data_set) const;
        
    public:
        trainer(double eta, uint64_t max_iterations);

        inline perceptron<T, Xdim, Ydim>
        train(matrix<T, Xdim, Ydim> const& train_data);
    };

    template<typename T, size_t Xdim, size_t Ydim>
    typename trainer<T, Xdim, Ydim>::matrix_view const& 
    trainer<T, Xdim, Ydim>::
    pick_data_random(matrix<T, Xdim, Ydim> const& data_set) const
    {
        size_t idx = _distr(_random_engine);
        return {data_set.begin(idx), data_set.end(idx)};
    }
}

#endif
