#include <algorithm>

#include "perceptron.hpp"

namespace perceptron
{
    template<typename T, size_t Xdim, size_t Ydim>
    void
    perceptron<T, Xdim, Ydim>::
    update_weight(vector<T, Ydim> const& correction)
    {
        weight += correction;
    }

    template<typename T, size_t Xdim, size_t Ydim>
    vector<T, Ydim>
    perceptron<T, Xdim, Ydim>::
    operator()(matrix<T, Xdim, Ydim> const& input_x) const
    {
        auto prod = blaze::dot(input_x, weight);

        std::transform(prod.begin(), prod.end(),
                       prod.begin(),
                       &perceptron<T, Xdim, Ydim>::sigmoid);

        return prod;
    }

    template<typename T, size_t Xdim, size_t Ydim>
    T
    perceptron<T, Xdim, Ydim>::
    operator()(vector<T, Xdim> const& input_x) const
    {
        return sigmoid(blaze::dot(input_x, weight));
    }
}
