#include <iostream>
#include <vector>
#include <thread>
#include <numeric>
#include <chrono>
#include <functional>
#include "thread_pool.hpp"  // Incluir tu implementación de thread_pool

// Estructura para representar una matriz dispersa
struct SparseMatrix {
    size_t rows, cols;
    std::vector<std::tuple<size_t, size_t, double>> values;  // (row, col, value)

    SparseMatrix(size_t r, size_t c) : rows(r), cols(c) {}

    // Insertar un valor no nulo en la matriz
    void insert(size_t row, size_t col, double value) {
        values.push_back({row, col, value});
    }
    // Obtener un valor de la matriz
    double get(size_t row, size_t col) const {
        for (const auto &[r, c, v] : values) {
            if (r == row && c == col) {
                return v;
            }
        }
        return 0.0;
    }
};

// Función para realizar la multiplicación de matrices dispersas
void sparseMatrixMultiplicationTask(
    const SparseMatrix &A, const SparseMatrix &B, SparseMatrix &C,
    size_t start_row, size_t end_row) {
    
    for (size_t i = start_row; i < end_row; ++i) {
        // Iterar sobre las filas de A
        for (const auto &[rowA, colA, valA] : A.values) {
            if (rowA == i) {
                // Iterar sobre las columnas de B para la multiplicación
                for (const auto &[rowB, colB, valB] : B.values) {
                    if (colA == rowB) {
                        // Multiplicar los valores no nulos
                        bool found = false;
                        for (auto &[rowC, colC, valC] : C.values) {
                            if (rowC == i && colC == colB) {
                                valC += valA * valB;
                                found = true;
                                break;
                            }
                        }
                        if (!found) {
                            C.insert(i, colB, valA * valB);
                        }
                    }
                }
            }
        }
    }
}

// Función para dividir el trabajo entre los hilos
void sparseMatrixMultiply(
    const SparseMatrix &A, const SparseMatrix &B, SparseMatrix &C,
    size_t num_threads) {

    thread_pool pool(num_threads);

    size_t rows_per_thread = A.rows / num_threads;
    size_t remaining_rows = A.rows % num_threads;

    size_t start_row = 0;

    for (size_t t = 0; t < num_threads; ++t) {
        size_t end_row = start_row + rows_per_thread + (t < remaining_rows ? 1 : 0);
        
        // Enviar el trabajo al thread pool
        pool.submit([=, &A, &B, &C] {
            sparseMatrixMultiplicationTask(A, B, C, start_row, end_row);
        });

        start_row = end_row;
    }

    // Asegurarse de que todos los hilos hayan terminado su trabajo
    pool.finish_work();
}

// Función para llenar aleatoriamente la matriz dispersa
void fillMatrixRandomly(SparseMatrix &matrix, double sparsity = 0.8, double value_range = 10.0) {
    srand(static_cast<unsigned int>(time(0)));  // Inicializar el generador de números aleatorios

    size_t non_zero_count = 0;

    // Calcular el número de elementos no nulos según la densidad deseada
    size_t total_elements = matrix.rows * matrix.cols;
    size_t num_non_zero = static_cast<size_t>(total_elements * (1.0 - sparsity));

    // Llenar la matriz con valores aleatorios
    while (non_zero_count < num_non_zero) {
        size_t row = rand() % matrix.rows;
        size_t col = rand() % matrix.cols;

        // Generar un valor aleatorio entre -value_range y +value_range
        double value = (rand() % static_cast<int>(value_range * 2)) - value_range;

        // Insertar el valor en la matriz si la posición no ha sido ocupada antes
        bool already_inserted = false;
        for (const auto &[r, c, v] : matrix.values) {
            if (r == row && c == col) {
                already_inserted = true;
                break;
            }
        }

        // Si no está ocupada, insertar el valor en la matriz
        if (!already_inserted) {
            matrix.insert(row, col, value);
            ++non_zero_count;
        }
    }
}

// Función para mostrar el uso correcto del programa
std::pair<size_t, size_t>
usage(int argc, char *argv[]) {
    if (argc != 3) {
        std::cerr << "Uso: " << argv[0] << " <dim> <hilos>" << std::endl;
        exit(1);
    }

    size_t dim = std::stoul(argv[1]);
    size_t threads = std::stoul(argv[2]);

    return std::make_pair(dim, threads);
}

int main(int argc, char *argv[]){

    // Parse command line arguments
    auto p = usage(argc, argv);
    size_t dim = p.first;
    size_t threads = p.second;

    // Definir matrices A y B (matrices dispersas)
    SparseMatrix A(dim, dim);
    fillMatrixRandomly(A, 0.8);

    for (const auto &[row, col, value] : A.values) {
        std::cout << "A[" << row << "][" << col << "] = " << value << std::endl;
    }
    
    SparseMatrix B(dim, dim);
    fillMatrixRandomly(B, 0.6);

    for (const auto &[row, col, value] : B.values) {
        std::cout << "B[" << row << "][" << col << "] = " << value << std::endl;
    }

    SparseMatrix C(dim, dim);

    auto start = std::chrono::steady_clock::now();

    // Llamar a la función de multiplicación de matrices dispersas usando el thread pool
    sparseMatrixMultiply(A, B, C, threads);

    auto stop = std::chrono::steady_clock::now();

    // Mostrar el resultado de la matriz C
    std::cout << "Resultado de la multiplicación de matrices dispersas:" << std::endl;
    for (const auto &[row, col, value] : C.values) {
        std::cout << "C[" << row << "][" << col << "] = " << value << std::endl;
    }
    std::cout << "Execution time: " <<
    std::chrono::duration_cast<std::chrono::milliseconds>(stop-start).count() << " ms." << std::endl;

    return 0;
}
