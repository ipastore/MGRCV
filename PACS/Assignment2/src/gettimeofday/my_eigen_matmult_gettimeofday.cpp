#include <iostream>
#include <ctime>
#include <Eigen/Dense>
#include <iomanip>
#include <sys/time.h>

using namespace std;

void calculateElapsedTime(const struct timeval& start, const struct timeval& end, const string& label) {
    double startTime = start.tv_sec + (start.tv_usec / 1000000.0);
    double endTime = end.tv_sec + (end.tv_usec / 1000000.0);
    cout << label << ": " << (endTime - startTime) << endl;
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
        
    Eigen::MatrixXd A = Eigen::MatrixXd::Random(N, N);
    Eigen::MatrixXd B = Eigen::MatrixXd::Random(N, N);
    Eigen::MatrixXd C(N, N);

    gettimeofday(&timestamp_end, NULL);
    calculateElapsedTime(timestamp_start, timestamp_end, "Initialization time");
    gettimeofday(&timestamp_start, NULL);

    C = A * B;

    gettimeofday(&timestamp_end, NULL);
    calculateElapsedTime(timestamp_start, timestamp_end, "Multiplication time");

}



