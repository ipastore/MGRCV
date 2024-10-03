#include <iostream>
#include <cstdlib>
#include <ctime>

using namespace std;

int main(int argc, char* argv[]) {

    if (argc != 2) {
        cerr << "Usage: " << argv[0] << " <matrix_size>" << endl;
        return 1;       
    }

    size_t MAX_SIZE = std::stoul(argv[1]);

    srand(time(0));

    // Stack-allocated 2D arrays
    double A[MAX_SIZE][MAX_SIZE];
    double B[MAX_SIZE][MAX_SIZE];
    double C[MAX_SIZE][MAX_SIZE] = {0};

    for (size_t i = 0; i < MAX_SIZE; i++) {
        for (size_t j = 0; j < MAX_SIZE; j++) {
            A[i][j] = static_cast<double>(rand()) / RAND_MAX * 10.0;
            B[i][j] = static_cast<double>(rand()) / RAND_MAX * 10.0;
        }
    }

    for (size_t i = 0; i < MAX_SIZE; i++) {
        for (size_t j = 0; j < MAX_SIZE; j++) {
            C[i][j] = 0;
            for (size_t k = 0; k < MAX_SIZE; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }

    return 0;
}
