#include <iostream>
#include <ctime>
#include <Eigen/Dense>

using namespace std;

int main (int argc, char* argv[]){

    if (argc != 2) {
        cerr << "Usage: " << argv[0] << " <matrix_size>" << endl;
        return 1;       
    }

    size_t N = std::stoul(argv[1]);
        
    Eigen::MatrixXd A = Eigen::MatrixXd::Random(N, N);
    Eigen::MatrixXd B = Eigen::MatrixXd::Random(N, N);
    Eigen::MatrixXd C(N, N);

    C = A * B;

    // cout <<  endl << "Matrix A:" << endl;
    // cout << A << endl;

    // cout << "Matrix B:" << endl;
    // cout << B << endl;

    // cout << "Matrix C:" << endl;
    // cout << C << endl;
}



