// pi_taylor_sequential.cc

#include <iomanip>
#include <iostream>
#include <climits>
#include <chrono>


// Allow to change the floating point type
using my_float = long double;

my_float pi_taylor(size_t steps) {

    my_float pi_aprox = 0;
    size_t n = 0;
    int sign =1;
		// No se si es igual o igual o menos. PREGUNTAR
    
    while(n <= steps){
	    pi_aprox += sign / (2.0 * n + 1.0);
	    sign = -sign;
	    n++;
    }

    return 4.0 * pi_aprox;
}

int main(int argc, const char *argv[]) {
    if (argc != 2) {
        std::cerr << "Invalid syntax: pi_taylor <steps>" << std::endl;
        return 1;
    }

    size_t steps = std::stoll(argv[1]);

    // Start time measurement
    auto start = std::chrono::high_resolution_clock::now();

    auto pi = pi_taylor(steps);

    // End time measurement
    auto end = std::chrono::high_resolution_clock::now();

    // Calculate the duration
    std::chrono::duration<double> duration = end - start;

    // Print the results
    std::cout << "For " << steps << ", pi value: "
        << std::setprecision(std::numeric_limits<long double>::digits10 + 1)
        << pi << std::endl;
    std::cout << "Execution time: " << duration.count() << " seconds" << std::endl;
}