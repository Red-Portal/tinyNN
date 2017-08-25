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
        return blaze::dot(input_x, weight);
    }

    template<typename T, size_t Xdim, size_t Ydim>
    T
    perceptron<T, Xdim, Ydim>::
    operator()(vector<T, Xdim> const& input_x) const
    {
        return blaze::dot(input_x, weight);
    }
}
