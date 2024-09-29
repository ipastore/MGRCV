#include <iostream>
#include <random>

/*
 * Developed by: 
 *     - David Padilla Orenga, NIA: 946874
 *     - Inacio Pastore Benaim, NIP: 920576
 * 
 * EXPLANATION:
 *      - The matrix is dynamically allocated in the heap using several new operators. This means that memory for the matrix rows
 *        and columns is explicitly allocated at runtime, and we must free this memory when it's no longer needed to avoid memory leaks.
 * 
 *      - In this code we have created matrix.array (into the struct). It is a pointer to a pointer (float**), which means it points to an
 *        array of pointers. This first (float**) points to an array of (float*) pointers representing each of the rows of the matrix 
 *        (matrix.array = new float*[matrix.rows]). 
 * 
 *      - Once created this first array of float pointers, for each row, we allocate memory for an array of float values (one for each column,
 *        corresponding to the elements of this row).
 * 
 *      - Since we're allocating memory dynamically using  several new[], we need to free that memory using delete[] once the program finish.
 *        Nevertheless, the deleting orders matters in order to avoid memory leaks. First we have to delete each of the float arrays corresponding
 *        to the elements/columns of each individual row. Secondly, we already can delete de array of pointers corresponding that contains the
 *        adresses the rows arrays. If this is done the oter way around, after deleting the float** we could not acces the floats* that were inside,
 *        leading to a memory leak.
 * 
 *      - PD: We've decided to create a Matrix struc to concentrate all the matrix information in one data type. We've also created some functions to
 *        have a more clear main. 
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
}

void check_dimensions(Matrix& matrix){
    if ((matrix.rows >= 5) || (matrix.columns >= 5)) {
        std::cerr << "Error: Rows/columns dimensions cannot be 5 or bigger." << std::endl;
        exit(1);
    } else if ((matrix.rows == 0) || (matrix.columns == 0)){
        std::cerr << "Error: Zero is not an accepted dimension." << std::endl;
        exit(1);
    }
}

void create_matrix(Matrix& matrix) {
    // Allocating memory for the rows
    matrix.array = new float*[matrix.rows];
    // Allocating memory for each column/element on a row
    for (size_t i = 0; i < matrix.rows; ++i) {
        matrix.array[i] = new float[matrix.columns];  
    }
}

void fill_matrix(Matrix& matrix, std::uniform_real_distribution<float> dis, std::mt19937 gen){

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
    // We have to delete first the columns arrays that are inside the rows array of pointers.
    // If we would do it the other way, we couldn't access the columsn arrays once deeleted the rows arrays and that would mean memory leaks.
    for (size_t n = 0; n < matrix.rows; ++n) {
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
    check_dimensions(M);
    create_matrix(M);
    fill_matrix(M, dis, gen);

    std::cout << std::endl << "The maximum and minimun values of the matrix are: " << M.max_value << " and " << M.min_value << std::endl;

    delete_matrix(M);
}
