#include <iostream>
#include <math.h>

using namespace std;

float triangleAreaFromVertices(float vertex_1_x, float vertex_1_y, 
								float vertex_2_x, float vertex_2_y, 
								float vertex_3_x, float vertex_3_y){
	
	return 0.5f * abs(vertex_1_x*(vertex_2_y-vertex_3_y) + vertex_2_x*(vertex_3_y-vertex_1_y) + vertex_3_x*(vertex_1_y-vertex_2_y));
}

int main() {
	// Now you can give the triangle the values on the code itself
	float vertex_1_x = 1.0f;
	float vertex_1_y = 0.0f;
	float vertex_2_x = 3.0f;
	float vertex_2_y = 0.0f;
	float vertex_3_x = 2.0f;
	float vertex_3_y = 2.0f;
	cout << "The area of the triangle is " 
				<< triangleAreaFromVertices(vertex_1_x, vertex_1_y, vertex_2_x, vertex_2_y, vertex_3_x, vertex_3_y) 
				<< endl;


    return 0;
}