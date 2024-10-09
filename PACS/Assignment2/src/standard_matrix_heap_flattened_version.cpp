#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>

using namespace std;

void fillMatrix(vector<double>& matrix, size_t N){
    for (size_t element = 0; element < N*N; element++){
        matrix[element] = static_cast<double>(rand()) / RAND_MAX * 10.0;
    }
}

void multiplyMatrices(const vector<double>& A,
                      const vector<double>& B,
                      vector<double>& C, size_t N) {
    for (size_t row = 0; row < N; row++) {
        for (size_t column = 0; column < N; column++) {
            for (size_t k = 0; k < N; k++) {
                C[row * N + column] += A[row * N + k] * B[k * N + column];
            }
        }
    }
}

int main (int argc, char* argv[]){

    if (argc != 2) {
        cerr << "Usage: " << argv[0] << " <matrix_size>" << endl;
        return 1;       
    }

    size_t N = std::stoul(argv[1]);
    srand(time(0)); 
    
    vector<double> A(N*N);
    vector<double> B(N*N);
    vector<double> C(N*N, 0);

    fillMatrix(A, N);
    fillMatrix(B, N);
    multiplyMatrices(A, B, C, N);

}
