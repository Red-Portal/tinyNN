#include <algorithm>
#include <iostream>

#include "trainer.hpp"
#include "trainable.hpp"

namespace tnn
{
    template<typename T, size_t InSize>
    trainer<T, InSize>::
    trainer(double eta, uint64_t max_iterations, bool verbose)
        :_eta(eta),
         _max_iterations(max_iterations),
         _trainee(nullptr),
         _error(),
         _distr(),
         _verbose(verbose)
    {
        std::random_device seed_gen;
        _random_engine.seed(seed_gen());

        _error.reserve(max_iterations);
    }

    template<typename T, size_t InSize>
    typename trainer<T, InSize>::column const& 
    trainer<T, InSize>::
    pick_data_random(matrix_dyn<T> const& data_set) 
    {
        auto idx = _distr(_random_engine);
        return colunm(data_set, idx);
    }

    template<typename T, size_t InSize>
    vector<T, InSize> 
    trainer<T, InSize>::
    col_to_vector(
        typename trainer<T, InSize>::column const& data_set) const
    {
        vector<T, InSize> input_vector;
        
        std::copy(data_set.begin(), data_set.end(),
                  input_vector.begin());
    }

    template<typename T, size_t InSize>
    T
    trainer<T, InSize>::
    predict(vector<T, InSize> const& data_set) const
    {

        T error = (*_trainee)(data_set);
        return error;
    }

    template<typename T, size_t InSize>
    perceptron<T, InSize>
    trainer<T, InSize>::
    train(matrix_dyn<T> const& train_data)
    {
        _distr.reset(0, train_data.rows() - 1);

        auto _perceptron =
            std::make_shared<perceptron<T, InSize>>();
        auto _trainable = _perceptron;

        if(_verbose)
            std::cout << "  iteration" 
                      << "  error" << std::endl;

        for(size_t it = 0; it < _max_iterations; ++it)
        {
            auto selected_data_range =
                col_to_vector(pick_data_random(train_data));

            T error = predict(selected_data_range);

            _error.push_back(error);
            auto correction = _eta * error * selected_data_range;
            _trainee->update_weight(correction);

            if(_verbose)
                std::cout << " -" << it << "     " <<
                    _error << std::endl;
        }
        return *_perceptron;
    }
}
