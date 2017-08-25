#include <chrono>
#include <random>
#include <functional>
#include <iostream>

#include <blaze/math/DynamicMatrix.h>

#include "perceptron.hpp"
#include "trainer.hpp"

namespace chrono = std::chrono;

template<typename T>
blaze::DynamicMatrix<T>
random_matrix(size_t x, size_t y, T bottom, T top)
{
    static std::random_device rand_div;
    static std::mt19937_64 mersenne(rand_div());
    static std::uniform_real_distribution<T> dist(bottom, top);

    auto mat = blaze::DynamicMatrix<float>(x, y);
    for(auto i = 0u; i < x; ++i)
    {
        for(auto j = 0u; j < y; ++j)
        {
            mat(i, j) = dist(mersenne);
        }
    }

    return mat;
}

int main()
{
    auto training_data = random_matrix(3, 100, -1, 1);
    
    auto start = std::chrono::steady_clock::now();
    
    auto end = std::chrono::steady_clock::now();

    std::cout << chrono::duration_cast<
        chrono::milliseconds>(end - start).count()
              << std::endl;

    return 0;
}
