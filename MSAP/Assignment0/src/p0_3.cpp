#include <iostream>
#include <math.h>

using namespace std;

float triangleAreaFromVertices(float v[]){
	return 0.5f * abs(v[0]*(v[3]-v[5]) + v[2]*(v[5]-v[1]) + v[4]*(v[1]-v[3]));
}

int main() {
	// Now you can give the triangle the values on the code itself
	float v[9] = {1.0f, 0.0f, 3.0f, 0.0f, 2.0f, 2.0f, 0, 3L};
	
	v[6] = triangleAreaFromVertices(v);
	cout << "The area of the triangle is " << v[6] << endl;
	cout << "The number of vertices is: " << v[7] << endl;
	// cout << "This is a: " << v[8] << endl;

	// cout << "The 7th position is: " << v[6] << endl;


	/* TO-DO: Once you tested the example above, create a new array of nine floats, 
	and try to assign the six coordinates of the triangle vertices, the triangle area, 
	the number of vertices of the triangle, and a string with the triangle name (e.g., “Triangle”) */ 
	
    return 0;
}