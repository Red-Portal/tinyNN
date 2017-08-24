#ifndef _LEARNING_DEVICE_HPP_
#define _LEARNING_DEVICE_HPP_

#include <stdint.h>
#inlcude <memory>

#include "perceptron.hpp"

namespace perceptron
{
    template<typename T, size_t Xdim, size_t Ydim>
    class trainer
    {
    private:
        double _eta;
        int64_t _max_iterations;
        std::unique_ptr<trainable> _perceptron;

    public:
        trainer(double eta, int64_t max_iterations);

        perceptron
        train();
    }
}

#endif
