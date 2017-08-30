#include <chrono>
#include <stdint.h>
#include <random>
#include <functional>
#include <iostream>

#include <blaze/math/StaticMatrix.h>
#include <blaze/math/DynamicVector.h>
#include <blaze/math/DynamicMatrix.h>

#include <tinynn/perceptron.hpp>
#include <tinynn/activation.hpp>

namespace chrono = std::chrono;

template<typename T>
blaze::DynamicMatrix<T>
make_train_data()
{
    auto dyn_matrix = blaze::DynamicMatrix<T>(4, 4);

    dyn_matrix(0, 0) = 0;
    dyn_matrix(0, 1) = 0;
    dyn_matrix(0, 2) = 1;
    dyn_matrix(0, 3) = 0;

    dyn_matrix(1, 0) = 0;
    dyn_matrix(1, 1) = 1;
    dyn_matrix(1, 2) = 1;
    dyn_matrix(1, 3) = 1;

    dyn_matrix(2, 0) = 1;
    dyn_matrix(2, 1) = 0;
    dyn_matrix(2, 2) = 1;
    dyn_matrix(2, 3) = 1;

    dyn_matrix(3, 0) = 1;
    dyn_matrix(3, 1) = 1;
    dyn_matrix(3, 2) = 1;
    dyn_matrix(3, 3) = 1;

    return dyn_matrix;
}

int main()
{
    auto dyn_matrix = make_train_data<double>();

    double eta = 2;
    auto iterations = 100;

    auto start = std::chrono::steady_clock::now();
    auto trainer = tnn::perceptron_trainer<double, 3>(
        eta, iterations, tnn::activation_function::sigmoid, true);

    auto model  = trainer.train(dyn_matrix);
    auto end = std::chrono::steady_clock::now();

    std::cout << "\ntrained perceptron with " << std::endl;
    std::cout << "learning rate of " << eta << std::endl;
    std::cout << iterations << " iterations" << std::endl;
    std::cout << "sigmoid activation function" << std::endl;
    std::cout << "time elapsed: "
              << chrono::duration_cast<chrono::microseconds>(
                  end - start).count()
              << "us"<< std::endl;


    std::cout << "\n in1 in2  out" << std::endl;
    std::cout << "--------------" << std::endl;

    std::vector<std::chrono::duration<long long, std::nano>> bench;

    {
        blaze::StaticVector<double, 3> test;
        test[0] = 0;
        test[1] = 0;
        test[2] = 1;

        auto start = std::chrono::steady_clock::now();
        auto result = model(test);
        auto end = std::chrono::steady_clock::now();

        std::cout << "  " << test[0] << "   " << test[1]
                  << "   " <<  result << std::endl;
        bench.push_back(end - start);
    }

    {
        blaze::StaticVector<double, 3> test;
        test[0] = 1;
        test[1] = 0;
        test[2] = 1;

        auto start = std::chrono::steady_clock::now();
        auto result = model(test);
        auto end = std::chrono::steady_clock::now();

        std::cout << "  " << test[0] << "   " << test[1] 
                  << "   " <<  result << std::endl;
        bench.push_back(end - start);
    }

    {
        blaze::StaticVector<double, 3> test;
        test[0] = 0;
        test[1] = 1;
        test[2] = 1;

        auto start = std::chrono::steady_clock::now();
        auto result = model(test);
        auto end = std::chrono::steady_clock::now();

        std::cout << "  " << test[0] << "   " << test[1]
                  << "   " <<  result << std::endl;
        bench.push_back(end - start);
    }

    {
        blaze::StaticVector<double, 3> test;
        test[0] = 1;
        test[1] = 1;
        test[2] = 1;

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
