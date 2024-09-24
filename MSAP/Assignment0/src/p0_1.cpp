#include <iostream>
#include <math.h>

float euclideanDistance(float x1, float y1, float x2, float y2){
	return sqrt(pow(x2-x1,2)+pow(y2-y1,2));
}

int main() {
	float x1, y1, x2, y2;
    std::cout << "Introduce x and y coordinates (sep. by space): ";
    std::cin >> x1 >> y1;
    std::cout << "Introduce (another) x and y coordinates (sep. by space): ";
    std::cin >> x2 >> y2;
    std::cout << "The distance between coordinates is " << euclideanDistance(x1,y1,x2,y2) << std::endl;
    return 0;
}