#include <algorithm>
#include <iostream>
#include <cmath>
#include <cassert>

#include "multi_layer_trainer.hpp"
#include "trainable.hpp"

namespace tnn
{
    template<typename T, size_t InSize>
    perceptron_trainer<T, InSize>::
    perceptron_trainer(std::vector<size_t> const& layer_setting,
                       double eta,
                       uint64_t max_iterations,
                       std::function<double(double)> const& activation_fun,
                       bool verbose,
                       bool history)
        :_eta(eta),
         _max_iterations(max_iterations),
         _trainee(nullptr),
         _history(),
         _distr(),
         _verbose(verbose),
         _save_history(history),
         _activation_func(activation_fun)
    {
        std::random_device seed_gen;
        _random_engine.seed(seed_gen());

        _history.reserve(max_iterations);
    }

    
    template<typename T, size_t InSize>
    typename perceptron_trainer<T, InSize>::separated_data_set
    perceptron_trainer<T, InSize>::
    separate_in_out(matrix_dyn<T> const& train_data) const
    {

    }

    template<typename T, size_t InSize>
    std::tuple<vector<T, InSize>, vector_dyn<T>>
    perceptron_trainer<T, InSize>::
    pick_random_data(separated_data_set const& set) 
    {
        auto idx = _distr(_random_engine);


    }

    template<typename T, size_t InSize>
    double
    perceptron_trainer<T, InSize>::
    accuracy_percent(perceptron<T, InSize> const& perceptron,
                     separated_data_set const& train_data) const
    {
        auto result = perceptron(std::get<0>(train_data));
        auto correct = std::get<1>(train_data);
        auto error_vec = vector_dyn<T>(correct - result);

        auto abs_error_vec = map(
            error_vec,
            [](T elem){ return std::abs(elem); });

        return std::accumulate(abs_error_vec.begin(),
                               abs_error_vec.end(), 0.0)
            / abs_error_vec.size() * 100.0;
    }

    // template<typename T, size_t InSize>
    // vector_dyn<T>
    // perceptron_trainer<T, InSize>::
    // predict(vector<T, InSize> const& input) const
    // {
    //     T predicted_result = (*_trainee)(input);
    //     return predicted_result;
    // }

    template<typename T, size_t InSize>
    void
    perceptron_trainer<T, InSize>::
    assert_train_data(matrix_dyn<T> const& train_data)
    {
        // auto rows = train_data.rows();
        // auto cols = train_data.columns();
        // (void)rows;

        // assert(cols < InSize + 2 &&
        //        "input dimension too small: the dimension should be InSize + 1(bias) + 1(answer).");
    }

    template<typename T, size_t InSize>
    perceptron_trainer<T, InSize>::history::
    history(vector<T, InSize>&& input_,
            vector<T, InSize>&& delta_,
            T answer_, T output_, T error_)
        : input(std::move(input_)),
          delta(std::move(delta_)),
          answer(answer_),
          output(output_),
          error(error_)
    {}

    template<typename T, size_t InSize>
    typename perceptron_trainer<T, InSize>::history
    perceptron_trainer<T, InSize>::
    make_history(vector<T, InSize>&& input, T output, T answer,
                 vector<T, InSize>&& delta, T error) const
    {
        auto container = typename perceptron_trainer<T, InSize>::history(
            std::move(input), std::move(delta), answer,
            output, error);

        return container;
    }

    template<typename T, size_t InSize>
    perceptron<T, InSize>
    perceptron_trainer<T, InSize>::
    train(matrix_dyn<T> const& train_data)
    {
        assert_train_data(train_data);

        auto train_data_separated = separate_in_out(train_data);

        _distr.param(
            std::uniform_int_distribution<uint32_t>::
            param_type(0, train_data.rows() - 1));

        auto _perceptron
            = std::make_unique<perceptron<T, InSize>>(
                _activation_func);

        _trainee = _perceptron.get();

        if(_verbose)
            std::cout << "  iteration" << "  error" << std::endl;

        for(size_t it = 0; it < _max_iterations; ++it)
        {
            auto [train_input, train_answer]
                = pick_random_data(train_data_separated);

            T result = predict(train_input);
            T error = train_answer - result;

            auto correction = _eta * error * train_input;
            _trainee->update_weight(correction);

            if(_save_history)
            {
                _history.push_back(
                    make_history(std::move(train_input),
                                 result, train_answer,
                                 std::move(correction), error));
            }

            if(_verbose)
                std::cout << " - " << it+1 << "         "
                          << error << std::endl;
        }
        
        if(_verbose)
            std::cout << "train over\n"
                      << "average error: "
                      << accuracy_percent(*_perceptron,
                                          train_data_separated)
                      << "%" << std::endl;

        return *_perceptron;
    }
}
