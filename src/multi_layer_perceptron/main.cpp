#include <vector>
#include <chrono>
#include <iostream>

#include <blaze/math/DynamicMatrix.h>

#include <tinynn/multi_layer.hpp>

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
    dyn_matrix(3, 3) = 0;

    return dyn_matrix;
}

int main()
{
    namespace activ_f = tnn::activation_function;

    auto layer_setting = std::vector<size_t>({2, 1});
    auto dyn_matrix = make_train_data<double>();

    double eta = 1;
    auto iterations = 1000000;

    auto start = std::chrono::steady_clock::now();
    auto trainer = tnn::multi_layer_trainer<double, 3>(
        layer_setting, eta, iterations, activ_f::sigmoid,
        activ_f::derived<activ_f::sigmoid>, false);

    auto model  = trainer.train(dyn_matrix);
    auto end = std::chrono::steady_clock::now();

    std::cout << "\ntrained perceptron with " << std::endl;
    std::cout << "learning rate of " << eta << std::endl;
    std::cout << iterations << " iterations" << std::endl;
    std::cout << "sigmoid activation function" << std::endl;
    std::cout << "time elapsed: "
              << std::chrono::duration_cast<
                  std::chrono::microseconds>(end - start).count()
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
                  << "   " <<  result[0] << std::endl;
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
                  << "   " <<  result[0] << std::endl;
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
                  << "   " <<  result[0] << std::endl;
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
                  << "   " << result[0] << std::endl;
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
