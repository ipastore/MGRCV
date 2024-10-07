#include <iostream>
#include <random>
#include <cassert>  // For assertions

/*
 * Developed by: 
 *     - David Padilla Orenga, NIA: 946874
 *     - Inacio Pastore Benaim, NIP: 920576
 * 
 * EXPLANATION OF ASSERTS:
 *      - The assert macro is used to check the correctness of the program. If the condition is true, the program continues to run normally.
 *      - First assert checks if the matrix dimensions are greater than zero. If the condition is false, the program stops and an error message is displayed.
 *      - Second assert checks if the matrix dimensions are less than 5. If the condition is false, the program stops and an error message is displayed.
 *      - Third assert checks if the matrix array is null for whatever reason. If the condition is false, the program stops and an error message is displayed.
 *      - Fourth assert checks if the matrix column pointer is null for whatever reason. If the condition is false, the program stops and an error message is displayed.
 *      - Fifth assert checks, in order to delete the matrix, if the matrix array is null for whatever reason. If the condition is false, the program stops and an error message is displayed.
 *      - Sixth assert checks, in order to delete the matrix, if the matrix column pointer is null for whatever reason. If the condition is false, the program stops and an error message is displayed.
 *
 *  */

struct Matrix
{
    float** array;
    size_t rows;
    size_t columns;
    float min_value = std::numeric_limits<float>::max();
    float max_value = std::numeric_limits<float>::lowest(); 
};

void get_matrix_dimensions(Matrix& matrix){
    std::cout << "Indicate rows and columns of the matrix:" << std::endl;
    std::cin >> matrix.rows >> matrix.columns;
    assert(matrix.rows > 0 && matrix.columns > 0 && "Matrix dimensions must be greater than zero.");
    assert(matrix.rows < 5 && matrix.columns < 5 && "Matrix dimensions must be less than 5.");
}

void create_matrix(Matrix& matrix) {
    // Allocating memory for the rows
    matrix.array = new float*[matrix.rows];
    assert(matrix.array != nullptr && "Memory allocation for matrix rows failed.");

    // Allocating memory for each column/element on a row
    for (size_t i = 0; i < matrix.rows; ++i) {
        matrix.array[i] = new float[matrix.columns];  
        assert(matrix.array[i] != nullptr && "Memory allocation for matrix columns failed.");
    }
}

void fill_matrix(Matrix& matrix, std::uniform_real_distribution<float> dis, std::mt19937 gen){
    assert(matrix.array != nullptr && "Matrix array is null. Cannot fill a null matrix.");
    std::cout << std::endl << "Used matrix is:" << std::endl << std::endl;

    for(size_t n = 0; n < matrix.rows; n++){
        for (size_t m = 0; m < matrix.columns; m++){
            matrix.array[n][m] = dis(gen);  
            std::cout << matrix.array[n][m] << ' ';

            // Checking min/max
            if (matrix.array[n][m] < matrix.min_value) matrix.min_value = matrix.array[n][m];
            if (matrix.array[n][m] > matrix.max_value) matrix.max_value = matrix.array[n][m];
        }
        std::cout << std::endl;
    }
}

void delete_matrix(Matrix& matrix) {
    assert(matrix.array != nullptr && "Matrix array is null. Cannot delete null pointers.");

    for (size_t n = 0; n < matrix.rows; ++n) {
        assert(matrix.array[n] != nullptr && "Matrix column pointer is null. Cannot delete null pointers.");
        delete[] matrix.array[n];
    }
    delete[] matrix.array;
}

int main() {

    std::random_device rd; 
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis {0.0f, 1.0f};
    Matrix M;

    get_matrix_dimensions(M);
    create_matrix(M);
    fill_matrix(M, dis, gen);

    std::cout << std::endl << "The maximum and minimun values of the matrix are: " << M.max_value << " and " << M.min_value << std::endl;

    delete_matrix(M);
}
