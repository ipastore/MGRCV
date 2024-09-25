#include <iostream>
#include <random>
using namespace std;

/*
Developed by: 
    - David Padilla Orenga, NIA: 946874
    - Inacio Pastore Benaim, NIP: 920576

Please write a program that finds the maximum and the minimum elements in a four by four float 2D
matrix. The first version, max_min_global_2d_matrix.cpp of the program should store the matrix in
the   data   section,   and   should   access   the   elements   with   a   pointer   to   float.   The   second   version,
max_min_heap_2d_matrix.cpp, should read two integer numbers from the command line and create
a matrix in the heap with those dimensions, always less than 5 rows and columns. To fill the matrix, use
the C++ standard library pseudo-random number generators ; e.g.:


The output of both programs should be:
The maximum and minimun values of the matrix are: XXX and YYY 
Notes: Could you ask yourself how the memory stores the matrix? Do you need to delete the matrix in
any case?
*/

int main() {

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis {0.0f, 1.0f};

    auto pseudo_random_float_value = dis(gen);

}
