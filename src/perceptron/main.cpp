#include <chrono>
#include <stdint.h>
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
        double count = 0;
        for(auto j = 0u; j < Ydim - 1; ++j)
        {
            auto number = dist(mersenne);
            matrix(i, j) = number;
            count += number;
        }

        if(count > 1.0)
            matrix(i, Ydim - 1) = 1;
        else
            matrix(i, Ydim - 1) = 0;
    }
    return matrix;
}

int main()
{
    auto matrix = random_matrix<100, 3, double>(0.0, 1.0);
    auto dyn_matrix = blaze::DynamicMatrix<double>(matrix);

    double eta = 0.5;
    auto iterations = 100;

    auto start = std::chrono::steady_clock::now();
    auto _trainer = tnn::trainer<double, 2>(eta, iterations, true);
    auto model  = _trainer.train(matrix);
    auto end = std::chrono::steady_clock::now();


    std::cout << "\ntrained perceptron with " << std::endl;
    std::cout << "learning rate of " << eta << std::endl;
    std::cout << iterations << " iterations" << std::endl;
    std::cout << "time elapsed: "
              << chrono::duration_cast<
        chrono::microseconds>(end - start).count()
              << "us"<< std::endl;


    std::cout << "\n in1 in2  out" << std::endl;
    std::cout << "--------------" << std::endl;

    std::vector<std::chrono::duration<long long, std::nano>> bench;

    {
        blaze::StaticVector<double, 2> test;
        test[0] = 0;
        test[1] = 0;

        auto start = std::chrono::steady_clock::now();
        auto result = model(test);
        auto end = std::chrono::steady_clock::now();

        std::cout << "  " << test[0] << "   " << test[1]
                  << "   " <<  result << std::endl;
        bench.push_back(end - start);
    }

    {
        blaze::StaticVector<double, 2> test;
        test[0] = 1;
        test[1] = 0;

        auto start = std::chrono::steady_clock::now();
        auto result = model(test);
        auto end = std::chrono::steady_clock::now();

        std::cout << "  " << test[0] << "   " << test[1] 
                  << "   " <<  result << std::endl;
        bench.push_back(end - start);
    }

    {
        blaze::StaticVector<double, 2> test;
        test[0] = 0;
        test[1] = 1;

        auto start = std::chrono::steady_clock::now();
        auto result = model(test);
        auto end = std::chrono::steady_clock::now();

        std::cout << "  " << test[0] << "   " << test[1]
                  << "   " <<  result << std::endl;
        bench.push_back(end - start);
    }

    {
        blaze::StaticVector<double, 2> test;
        test[0] = 1;
        test[1] = 1;

        auto start = std::chrono::steady_clock::now();
        auto result = model(test);
        auto end = std::chrono::steady_clock::now();

        std::cout << "  " << test[0] << "   " << test[1]
                  << "   " << result << std::endl;
        bench.push_back(end - start);
    }

    std::cout << "\n";
    for(auto i = 0; i < 4; ++i)
    {
        std::cout << "eval time for " << i
                  << "th evaluation: "
                  << bench[i].count()
                  << "ns"<< std::endl;
    }

    return 0;
}
