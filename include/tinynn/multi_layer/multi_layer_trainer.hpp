#ifndef _MULTI_LAYER_TRAINER_HPP_
#define _MULTI_LAYER_TRAINER_HPP_

#include <vector>
#include <utility>
#include <functional>
#include <random>

#include "trainable.hpp"
#include "multi_layer_perceptron.hpp"

namespace tnn
{
    template<typename T, size_t InSize>
    class multi_layer_trainer
    {
        using separated_data_set =
            std::tuple<vector<T, InSize>, vector_dyn<T>>;
    public:
        struct history
        {
            inline history(vector<T, InSize>&& input_,
                           vector_dyn<T>&& answer_,
                           vector_dyn<T>&& output_,
                           T error_);
            vector<T, InSize> input;
            vector_dyn<T> answer;
            vector_dyn<T> output;
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
        std::function<double(double)> _activation_func_derived;
        std::vector<size_t> _layer_setting;

        inline std::vector<vector_dyn<T>>
        forward_propagate(vector<T, InSize> const& input) const;

        inline void
        backward_propagate(
            std::vector<vector_dyn<T>> const& output_per_layer,
            vector_dyn<T> const& answer);

        inline double
        calculate_error(vector_dyn<T> const& error) const;

        inline history
        make_history(vector<T, InSize>&& input,
                     vector_dyn<T>&& output,
                     vector_dyn<T>&& answer,
                     T error) const;

        inline void
        assert_train_data(matrix_dyn<T> const& input);

        inline matrix_dyn<T>
        vector_to_matrix(
            vector_dyn<T, blaze::rowVector> const& vec) const;

        inline double
        accuracy_percent(
            multi_layer_perceptron<T, InSize> const& perceptron,
            matrix_dyn<T> const& train_data) const;

        inline std::vector<separated_data_set>
        separate_in_out(matrix_dyn<T> const& train_data) const;

        inline std::tuple<vector<T, InSize>, vector_dyn<T>>
        pick_random_data(
            std::vector<separated_data_set> const& data_set);

    public:
        inline multi_layer_trainer(
            std::vector<size_t> const& layer_setting,
            double eta, uint64_t max_iterations,
            std::function<double(double)> const& activation_fun,
            std::function<double(double)> const& activation_derived,
            bool verbose = true,
            bool history = false);

        inline multi_layer_perceptron<T, InSize>
        train(matrix_dyn<T> const& train_data);

    };
}

#include "multi_layer_trainer.tpp"

#endif
