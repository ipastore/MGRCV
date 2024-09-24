#include <iostream>
#include <math.h>

struct Vertex2 {
	float x;
	float y;
	std::string name;
};

class Triangle{
	public:
		Vertex2 v1;
		Vertex2 v2;
		Vertex2 v3;
		int number_of_vertices;
		std::string name;
};

int main() {
	// Create an empty object
	Triangle my_triangle;
	// Assign each coordinate
	Vertex2 v1;
	v1.x = 1.0f;
	v1.y = 0.0f;
	v1.name = "Bottom-left corner";
	my_triangle.v1 = v1;
	Vertex2 v2;
	v2.x = 3.0f;
	v2.y = 0.0f;
	v2.name = "Bottom-right corner";
	my_triangle.v2 = v2;
	Vertex2 v3;
	v3.x = 2.0f;
	v3.y = 2.0f;
	v3.name = "Top corner";
	my_triangle.v3 = v3;
	// Assign name
	my_triangle.name = "Umbrella";
	// Assign number of vertices
	my_triangle.number_of_vertices = 3;
	// Show info
	std::cout << "My " + my_triangle.name << 
					" has " << my_triangle.number_of_vertices << 
					" vertices in coordinates " <<
					my_triangle.v1.x << "," << my_triangle.v1.y << " " <<
					my_triangle.v2.x << "," << my_triangle.v2.y << " " <<
					my_triangle.v3.x << "," << my_triangle.v3.y << std::endl;
					
					
					
	// Same example as above but using pointer types
	// This creates a variable which is a pointer to Triangle
	Triangle* my_triangle_pointer =  new Triangle(); 
	/* This allocates memory for a Triangle object and assigns 
	the resulting memory address to the pointer variable */
	// my_triangle_pointer = new Triangle(); 
	my_triangle_pointer->v1 = v1;
	my_triangle_pointer->v2 = v2;
	my_triangle_pointer->v3 = v3;
	my_triangle_pointer->number_of_vertices = 3;
	
	my_triangle_pointer->name = "Umbrella (created with a pointer)";
	
	
	std::cout << "My " + my_triangle_pointer->name << 
					" has " << my_triangle_pointer->number_of_vertices << 
					" vertices in coordinates " <<
					my_triangle_pointer->v1.x << "," << my_triangle_pointer->v1.y << " " <<
					my_triangle_pointer->v2.x << "," << my_triangle_pointer->v2.y << " " <<
					my_triangle_pointer->v3.x << "," << my_triangle_pointer->v3.y << std::endl;
	


    return 0;
}