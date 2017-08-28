#ifndef _LAYER_HPP_
#define _LAYER_HPP_

#include "../perceptron/perceptron_impl.hpp"

namespace tnn
{
    class layer
    {
    private:
        std::vector<perceptron> _nodes;
    public:
        layer(size_t perceptron_number);
    }
}

#endif
