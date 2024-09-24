#include <iostream>
#include <math.h>

struct Vertex2 {
	float x;
	float y;
	std::string name;
};

class Triangle{
	private:
		Vertex2 v1;
		Vertex2 v2;
		Vertex2 v3;
		int number_of_vertices;
		std::string name;
	public:
		// Setters
		void set_name(std::string name){
			this->name = name;
		}
		void set_number_of_vertices(int number_of_vertices){
			if (number_of_vertices != 3){
				std::cerr << "This does not look like a triangle..." << std::endl;
			}
			this->number_of_vertices = number_of_vertices;
		}
		void set_vertices(Vertex2 v1, Vertex2 v2, Vertex2 v3){
			this->v1 = v1;
			this->v2 = v2;
			this->v3 = v3;
		}
		// Getters
		std::string get_name(){
			return this->name;
		}
		int get_number_of_vertices(){
			return this->number_of_vertices;
		}
		Vertex2 get_v1(){
			return this->v1;
		}
		Vertex2 get_v2(){
			return this->v2;
		}
		Vertex2 get_v3(){
			return this->v3;
		}
};

int main() {
	// Create an empty object
	Triangle my_triangle;
	// Assign each coordinate
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
	my_triangle.set_vertices(v1, v2, v3);
	// Assign name
	my_triangle.set_name("Umbrella");
	// Assign number of vertices
	my_triangle.set_number_of_vertices(3);
	// Show info
	std::cout << "My " + my_triangle.get_name() << 
					" has " << my_triangle.get_number_of_vertices() << 
					" vertices in coordinates " <<
					my_triangle.get_v1().x << "," << my_triangle.get_v1().y << " " <<
					my_triangle.get_v2().x << "," << my_triangle.get_v2().y << " " <<
					my_triangle.get_v3().x << "," << my_triangle.get_v3().y << std::endl;

    return 0;
}