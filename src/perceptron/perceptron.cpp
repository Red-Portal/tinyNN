#include "perceptron.hpp"

namespace perceptron
{
    template<typename T, size_t Xdim, size_t Ydim>
    void
    perceptron::
    update_weight(vector<T, Ydim> const& value)
    {
        weight += value;
    }

    template<typename T, size_t Xdim, size_t Ydim>
    vector<T, Ydim>
    perceptron::
    operator()(vector<T, Xdim> const& input_x) const
    {
        return blaze::dot(input_x, weight);
    }
}
