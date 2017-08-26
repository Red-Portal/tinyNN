#ifndef _TRAINER_HPP_
#define _TRAINER_HPP_

#include <stdint.h>
#include <vector>
#include <memory>
#include <random>

#include <blaze/math/StaticMatrix.h>
#include <blaze/math/StaticVector.h>
#include <blaze/math/DynamicMatrix.h>
#include <blaze/math/Column.h>

#include "trainable.hpp"
#include "perceptron.hpp"

namespace tnn
{
    template<typename T, size_t Xdim, size_t Ydim>
    using matrix = blaze::StaticMatrix<T, Xdim, Ydim>;

    template<typename T, size_t InSize>
    class trainer
    {
    private:
        double _eta;
        uint64_t _max_iterations;
        trainable<T, InSize>* _trainee;
        std::vector<T> _error;
        std::mt19937 _random_engine;
        std::uniform_int_distribution<uint32_t> _distr;
        bool _verbose;

        inline double
        accuracy_percent(
            perceptron<T, InSize> const& perceptron,
            std::tuple<matrix_dyn<T>, vector_dyn<T>> const& train_data) const;

        inline std::tuple<matrix_dyn<T>, vector_dyn<T>>
        separate_in_out(matrix_dyn<T> const& train_data) const;

        inline std::tuple<vector<T, InSize>, T>
        pick_data_random(
            std::tuple<matrix_dyn<T>, vector_dyn<T>> const& data_set);

        inline T predict(vector<T, InSize> const& data_set) const;
        
    public:
        inline trainer(double eta, uint64_t max_iterations,
                       bool verbose = true);

        inline perceptron<T, InSize>
        train(matrix_dyn<T> const& train_data);
    };
}
#include "trainer.tcc"

#endif
