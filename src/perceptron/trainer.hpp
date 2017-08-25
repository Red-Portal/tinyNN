#ifndef _LEARNING_DEVICE_HPP_
#define _LEARNING_DEVICE_HPP_

#include <stdint.h>
#include <vector>
#include <memory>
#include <random>

#include <blaze/math/StaticMatrix.h>
#include <blaze/math/Column.h>

#include "trainable.hpp"
#include "perceptron.hpp"

namespace perceptron
{
    template<typename T, size_t Xdim, size_t Ydim>
    using matrix = blaze::StaticMatrix<T, Xdim, Ydim>;

    template<typename T, size_t Ydim>
    using vector = blaze::StaticVector<T, Ydim>;

    template<typename T, size_t Xdim, size_t Ydim>
    class trainer
    {
        using column =
            typename blaze::Column<matrix<T, Xdim, Ydim>>;
    private:
        bool _verbose;
        double _eta;
        uint64_t _max_iterations;
        std::shared_ptr<perceptron<T, Xdim, Ydim>> _trainee;
        std::vector<T> _error;
        std::mt19937 _random_engine;
        std::uniform_int_distribution<uint32_t> _distr;

        inline column const& 
        pick_data_random(matrix<T, Xdim, Ydim> const& data_set);

        inline vector<T, Xdim>
        col_to_vector(column const& vec) const;

        T predict(vector<T, Xdim> const& data_set) const;
        
    public:
        trainer(double eta, uint64_t max_iterations,
                bool verbose = true);

        perceptron<T, Xdim, Ydim>
        train(matrix<T, Xdim, Ydim> const& train_data);
    };

    template<typename T, size_t Xdim, size_t Ydim>
    typename trainer<T, Xdim, Ydim>::column const& 
    trainer<T, Xdim, Ydim>::
    pick_data_random(matrix<T, Xdim, Ydim> const& data_set) 
    {
        auto idx = _distr(_random_engine);
        return colunm(data_set, idx);
    }

    template<typename T, size_t Xdim, size_t Ydim>
    vector<T, Xdim>
    trainer<T, Xdim, Ydim>::
    col_to_vector(
        typename trainer<T, Xdim, Ydim>::column const& data_set) const
    {
        vector<T, Xdim> input_vector;
        
        std::copy(data_set.begin(), data_set.end(),
                  input_vector.begin());
    }
}

#endif
