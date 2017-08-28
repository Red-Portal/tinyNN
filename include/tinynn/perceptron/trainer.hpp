#ifndef _TRAINER_HPP_
#define _TRAINER_HPP_

#include <stdint.h>
#include <vector>
#include <memory>
#include <random>

#include <blaze/math/StaticMatrix.h>
#include <blaze/math/StaticVector.h>

#include "trainable.hpp"
#include "perceptron_impl.hpp"

namespace tnn
{
    template<typename T, size_t Xdim, size_t Ydim>
    using matrix = blaze::StaticMatrix<T, Xdim, Ydim>;

    template<typename T, size_t Size>
    using vector = blaze::StaticVector<T, Size>;

    template<typename T, size_t InSize>
    class trainer
    {
        using separated_data_set =
            std::tuple<matrix_dyn<T>, vector_dyn<T>>;
    public:
        struct history
        {
            inline history(vector<T, InSize>&& input_,
                           vector<T, InSize>&& delta_,
                           T answer_, T output_, T error_);

            vector<T, InSize> input;
            vector<T, InSize> delta;
            T answer;
            T output;
            T error;
        };

    private:
        double _eta;
        uint64_t _max_iterations;
        trainable<T, InSize>* _trainee;
        std::vector<history> _history;
        std::mt19937 _random_engine;
        std::uniform_int_distribution<uint32_t> _distr;
        bool _verbose;
        bool _save_history;
        std::function<double(double)> _activation_func;

        inline history
        make_history(vector<T, InSize>&& input, T output, T answer,
                     vector<T, InSize>&& delta, T error) const;

        inline void
        assert_train_data(matrix_dyn<T> const& input);

        inline double
        accuracy_percent(
            perceptron<T, InSize> const& perceptron,
            separated_data_set const& train_data) const;

        inline separated_data_set
        separate_in_out(matrix_dyn<T> const& train_data) const;

        inline std::tuple<vector<T, InSize>, T>
        pick_random_data(
            separated_data_set const& data_set);

        inline T
        predict(vector<T, InSize> const& data_set) const;
        
    public:
        inline trainer(
            double eta, uint64_t max_iterations,
            std::function<double(double)> const& activation_fun,
            bool verbose = true,
            bool history = false);

        inline perceptron<T, InSize>
        train(matrix_dyn<T> const& train_data);

    };
}
#include "trainer.tcc"

#endif
