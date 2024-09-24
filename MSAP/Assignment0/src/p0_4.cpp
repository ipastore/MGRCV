#include <iostream>
#include <math.h>

struct Vertex2 {
	float x;
	float y;
	std::string name;
};


float triangleAreaFromVertices(Vertex2 v1, Vertex2 v2, Vertex2 v3){
	return 0.5f * abs(v1.x*(v2.y-v3.y) + v2.x*(v3.y-v1.y) + v3.x*(v1.y-v2.y));
}

int main() {
	// Now you can give the triangle the values on the code itself
	Vertex2 v1;
	v1.x = 1.0f;
	v1.y = 0.0f;
	v1.name = "Bottom-left corner";
	Vertex2 v2;
	v2.x = 3.0f;
	v2.y = 0.0f;
	v2.name = "Bottom-right corner";
	Vertex2 v3;
	v3.x = 2.0f;
	v3.y = 2.0f;
	v3.name = "Top corner";

	std::cout << "The area of the triangle is " << triangleAreaFromVertices(v1, v2, v3) << std::endl;
	  
    return 0;
}