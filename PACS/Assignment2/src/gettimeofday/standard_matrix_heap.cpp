#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <iomanip>
#include <sys/time.h>

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


void calculateElapsedTime(const struct timeval& start, const struct timeval& end) {
    double startTime = start.tv_sec + (start.tv_usec / 1000000.0);
    double endTime = end.tv_sec + (end.tv_usec / 1000000.0);
    cout << "Elapsed time: " << (endTime - startTime) << " seconds" << endl;
}


int main (int argc, char* argv[]){

    struct timeval timestamp_start;
    struct timeval timestamp_end;

    gettimeofday(&timestamp_start, NULL);

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

    gettimeofday(&timestamp_end, NULL);
    calculateElapsedTime(timestamp_start, timestamp_end);

    gettimeofday(&timestamp_start, NULL);
    multiplyMatrices(A, B, C, N);
    gettimeofday(&timestamp_end, NULL);
    calculateElapsedTime(timestamp_start, timestamp_end);

}
