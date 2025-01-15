#include <vector>
#include <future>
#include <iostream>
#include "matrix.hpp" // Suponiendo que tienes una clase matrix implementada

using fmatrix = matrix<float>;

// Versión paralela de la multiplicación de matrices usando std::async
fmatrix async_col_matrix_multiply(const fmatrix& a, const fmatrix& b) {
    if (a.cols() != b.rows()) {
        throw std::invalid_argument("Las dimensiones de las matrices no son compatibles para la multiplicación.");
    }

    fmatrix c(a.rows(), b.cols()); // Matriz resultado
    size_t num_threads = std::thread::hardware_concurrency(); // Número de hilos soportados por el sistema

    if (num_threads == 1) { // Si solo hay un hilo disponible, usar implementación secuencial
        for (size_t i = 0; i < a.rows(); ++i) {
            for (size_t j = 0; j < b.cols(); ++j) {
                float val = 0.0f;
                for (size_t k = 0; k < a.cols(); ++k) {
                    val += a(i, k) * b(k, j);
                }
                c(i, j) = val;
            }
        }
        return c;
    }

    // Si hay múltiples hilos, dividir el trabajo por columnas
    std::vector<std::future<void>> futures;

    for (size_t j = 0; j < b.cols(); ++j) {
        futures.emplace_back(std::async(std::launch::async, [&, j]() {
            for (size_t i = 0; i < a.rows(); ++i) {
                float val = 0.0f;
                for (size_t k = 0; k < a.cols(); ++k) {
                    val += a(i, k) * b(k, j);
                }
                c(i, j) = val;
            }
        }));
    }

    for (auto& f : futures) {
        f.get(); // Esperar a que cada tarea termine
    }

    return c;
}

// Ejemplo de uso
int main() {
    // Crear matrices de ejemplo
    fmatrix a(2, 3); // 2 filas, 3 columnas
    fmatrix b(3, 2); // 3 filas, 2 columnas

    // Inicializar valores para las matrices
    a(0, 0) = 1; a(0, 1) = 2; a(0, 2) = 3;
    a(1, 0) = 4; a(1, 1) = 5; a(1, 2) = 6;

    b(0, 0) = 7; b(0, 1) = 8;
    b(1, 0) = 9; b(1, 1) = 10;
    b(2, 0) = 11; b(2, 1) = 12;

    // Realizar la multiplicación de matrices
    fmatrix c = async_col_matrix_multiply(a, b);

    // Mostrar el resultado
    for (size_t i = 0; i < c.rows(); ++i) {
        for (size_t j = 0; j < c.cols(); ++j) {
            std::cout << c(i, j) << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}