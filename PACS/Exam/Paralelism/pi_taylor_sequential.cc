#include <iomanip>
#include <iostream>
#include <limits>
#include <chrono>   

// Allow to change the floating point type
using my_float = long double;

my_float pi_taylor(size_t steps) {
    my_float pi = 0.0;
    my_float sign = 1.0;

    // Serie de Gregory-Leibniz para aproximar pi
    for (size_t i = 0; i < steps; ++i) {
        my_float term = sign / (2 * i + 1);
        pi += term;
        sign = -sign;
    }

    return pi * 4.0;
}

int main(int argc, const char *argv[]) {

    // read the number of steps from the command line
    if (argc != 2) {
        std::cerr << "Invalid syntax: pi_taylor <steps>" << std::endl;
        exit(1);

    }
    auto start = std::chrono::high_resolution_clock::now();

    size_t steps = std::stoll(argv[1]);
    auto pi = pi_taylor(steps);

    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

    std::cout << "For " << steps << ", pi value: "
        << std::setprecision(std::numeric_limits<my_float>::digits10 + 1)
        << pi << " cost " << duration.count() << " microseconds" << std::endl;
}
