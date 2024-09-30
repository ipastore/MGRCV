#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>

using namespace std;

void fillMatrix(vector<vector<double>>& matrix, size_t N){
    for (size_t row = 0; row < N; row++){
        for (size_t column = 0; column < N; column++){
                matrix[row][column] = rand() % 10;
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
void printMatrix(const vector<vector<double>>& matrix, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            cout << matrix[i][j] << " ";
        }
        cout << endl;
    }
}

int main (){
    size_t N = 5;
    srand(time(0)); 
    
    vector<vector<double>> A(N, vector<double>(N));
    vector<vector<double>> B(N, vector<double>(N));
    vector<vector<double>> C(N, vector<double>(N, 0));

    fillMatrix(A, N);
    fillMatrix(B, N);

    cout <<  endl << "Matrix A:" << endl;
    printMatrix(A, N);

    cout << "Matrix B:" << endl;
    printMatrix(B, N);

    multiplyMatrices(A, B, C, N);

    cout << "Matrix C:" << endl;
    printMatrix(C, N);

}