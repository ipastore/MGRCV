#include <iostream>
#include <ctime>
#include <Eigen/Dense>
#include <iomanip>

using namespace std;

int main (int argc, char* argv[]){

    clock_t timer_1 = clock();

    if (argc != 2) {
        cerr << "Usage: " << argv[0] << " <matrix_size>" << endl;
        return 1;       
    }

    size_t N = std::stoul(argv[1]);
        
    Eigen::MatrixXd A = Eigen::MatrixXd::Random(N, N);
    Eigen::MatrixXd B = Eigen::MatrixXd::Random(N, N);
    Eigen::MatrixXd C(N, N);

    C = A * B;

    timer_1 = clock() - timer_1;
    cout << "Seconds of execution: " << fixed << setprecision(4) << ((float)timer_1)/CLOCKS_PER_SEC << endl;
}



