#include <chrono>
#include <random>
#include <functional>
#include <iostream>

#include <blaze/math/StaticMatrix.h>
#include <blaze/math/DynamicVector.h>
#include <blaze/math/DynamicMatrix.h>

#include "perceptron.hpp"
#include "trainer.hpp"

namespace chrono = std::chrono;

template<size_t Xdim, size_t Ydim, typename T>
blaze::StaticMatrix<T, Xdim, Ydim>
random_matrix(T bottom, T top)
{
    static std::random_device rand_div;
    static std::mt19937_64 mersenne(rand_div());
    static std::uniform_real_distribution<T> dist(bottom, top);

    blaze::StaticMatrix<T, Xdim, Ydim> matrix{}; 
    for(auto i = 0u; i < Xdim; ++i)
    {
        int count = 0;
        for(auto j = 0u; j < Ydim - 1; ++j)
        {
            auto number = dist(mersenne);
            matrix(i, j) = number;
            count += number;
        }

        if(count > 0)
            matrix(i, Ydim - 1) = 1;
        else
            matrix(i, Ydim - 1) = 0;
    }
    return matrix;
}

int main()
{
    auto matrix = random_matrix<3, 100, double>(-10.0, 10.0);
    auto dyn_matrix = blaze::DynamicMatrix<double>(matrix);

    auto start = std::chrono::steady_clock::now();
    auto _trainer = tnn::trainer<double, 3>(0.2, 100, true);
    auto model  = _trainer.train(matrix);
    auto end = std::chrono::steady_clock::now();

    blaze::StaticVector<double, 3> test;
    test[0] = 1;
    test[0] = 1;
    test[0] = 1;
    auto result  = model(test);

    std::cout << "result: " << result << std::endl;

    std::cout << chrono::duration_cast<
        chrono::milliseconds>(end - start).count()
              << std::endl;

    return 0;
}
