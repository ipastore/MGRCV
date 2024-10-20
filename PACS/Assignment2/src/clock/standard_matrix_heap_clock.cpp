#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <iomanip>

using namespace std;

void fillMatrix(vector<vector<double>>& matrix, size_t N){
    for (size_t row = 0; row < N; row++){
        for (size_t column = 0; column < N; column++){
            matrix[row][column] = static_cast<double>(rand()) / RAND_MAX * 2.0 - 1.0; // Generate random numbers between -1 and 1
        }
    }
}

void multiplyMatrices(const vector<vector<double>>& A,
                      const vector<vector<double>>& B,
                      vector<vector<double>>& C, size_t N){

    for (size_t row = 0; row < N; row++) {
        for (size_t column = 0; column < N; column++) {
            for (size_t element = 0; element < N; element++){
                C[row][column] += A[row][element] * B[element][column];
            }
        }  
    }   
}

int main (int argc, char* argv[]){

    // Start clock
    clock_t timer_1 = clock();

    if (argc != 2) {
        cerr << "Usage: " << argv[0] << " <matrix_size>" << endl;
        return 1;       
    }

    size_t N = std::stoul(argv[1]);
    vector<vector<double>> A(N, vector<double>(N));
    vector<vector<double>> B(N, vector<double>(N));
    vector<vector<double>> C(N, vector<double>(N, 0));

    // Fill matrices with random data between -1 and 1 (same as Eigen)
    fillMatrix(A, N);
    fillMatrix(B, N);
    multiplyMatrices(A, B, C, N);

    // Stop clock and display time
    timer_1 = clock() - timer_1;
    cout << "Seconds of execution: " << fixed << setprecision(4) << ((float)timer_1)/CLOCKS_PER_SEC << endl;

    return 0;
}
