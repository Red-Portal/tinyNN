#include <algorithm>
#include <iostream>

#include "trainer.hpp"
#include "trainable.hpp"

namespace perceptron
{
    template<typename T, size_t Xdim, size_t Ydim>
    trainer<T, Xdim, Ydim>::
    trainer(double eta, uint64_t max_iterations, bool verbose)
        :_eta(eta),
         _max_iterations(max_iterations),
         _trainee(nullptr),
         _error(),
         _distr(0, Ydim - 2),
         _verbose(verbose)
    {
        std::random_device seed_gen;
        _random_engine.seed(seed_gen());

        _error.reserve(max_iterations);
    }


    template<typename T, size_t Xdim, size_t Ydim>
    T
    trainer<T, Xdim, Ydim>::
    predict(vector<T, Xdim> const& data_set) const
    {

        T error = (*_trainee)(data_set);
        return error;
    }

    template<typename T, size_t Xdim, size_t Ydim>
    perceptron<T, Xdim, Ydim>
    trainer<T, Xdim, Ydim>::
    train(matrix<T, Xdim, Ydim> const& train_data)
    {
        auto _perceptron =
            std::make_shared<perceptron<T, Xdim, Ydim>>();
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
