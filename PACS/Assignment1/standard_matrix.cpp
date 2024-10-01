#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>

using namespace std;

void fillMatrix(vector<vector<double>>& matrix, size_t N){
    for (size_t row = 0; row < N; row++){
        for (size_t column = 0; column < N; column++){
                matrix[row][column] = static_cast<double>(rand()) / RAND_MAX * 10.0;
        }
    }
}

void multiplyMatrices(const vector<vector<double>>& A,
                      const vector<vector<double>>& B,
                      vector<vector<double>>& C, size_t N){

    for (size_t row = 0; row < N; row++) {
        for (size_t column = 0; column < N; column++) {
            for (size_t element = 0; element < N; element++)
            {
                C[row][column] += A[row][element] * B[element][column];
            }
        }  
    }   
}

// Function to print a matrix (for testing purposes)
// void printMatrix(const vector<vector<double>>& matrix, int N) {
//     for (int i = 0; i < N; i++) {
//         for (int j = 0; j < N; j++) {
//             cout << matrix[i][j] << " ";
//         }
//         cout << endl;
//     }
// }

int main (int argc, char* argv[]){

    if (argc != 2) {
        cerr << "Usage: " << argv[0] << " <matrix_size>" << endl;
        return 1;       
    }

    size_t N = std::stoul(argv[1]);
    srand(time(0)); 
    
    vector<vector<double>> A(N, vector<double>(N));
    vector<vector<double>> B(N, vector<double>(N));
    vector<vector<double>> C(N, vector<double>(N, 0));

    fillMatrix(A, N);
    fillMatrix(B, N);
    multiplyMatrices(A, B, C, N);

    // cout <<  endl << "Matrix A:" << endl;
    // printMatrix(A, N);

    // cout << "Matrix B:" << endl;
    // printMatrix(B, N);

    // cout << "Matrix C:" << endl;
    // printMatrix(C, N);

}
