#include <iostream>
#include <memory>


struct point
{
    int x, y;
    bool not_zero() const { return x!=0 || y!=0; };
};


struct point_node
{
    point p;
    // point_node* next;
    std::unique_ptr<point_node> next; 
    // point_node(const point& other_p, point_node* other_next) : p(other_p), next(other_next) {}
    point_node(const point& other_p) : p(other_p), next(nullptr) {}
};


point* copy_point(const point& p)
{
    point* point_ptr = new point(p);
    if (p.x > 1)
        return nullptr;
    return point_ptr;
}


int main()
{
    const size_t n = 4;
    point points[n]={ {0, 0}, {1, 1}, {0, 0}, {1, 1} };

    // save points whose coordinates are not (0, 0)
    std::unique_ptr<point_node> list = nullptr;

    for(const auto& p: points) {
        std::unique_ptr<point_node> node = std::make_unique<point_node>(p);
        
        if(p.not_zero()) {
            if(list == nullptr) {
                list = node;
            } else {
                point_node* iter=list;
                while(iter->next != nullptr) {
                    iter = iter->next;
                }
                iter->next = node;
            }
        }
    }
    
    // print non-zero nodes
    for(point_node* iter = list; iter != nullptr; iter = iter->next) {
        std::cout << "(x, y): " << iter->p.x << ", " << iter->p.y << std::endl;
    }
}

int main()
{
    const size_t n = 4;
    point points[n]={ {0, 0}, {1, 1}, {0, 0}, {1, 1} };

    // save points whose coordinates are not (0, 0)
    std::unique_ptr<point_node> list = nullptr;

    for(const auto& p: points) {
        auto node = std::make_unique<point_node>(p);  // Create a new point_node using std::unique_ptr
        
        if(p.not_zero()) {
            if(list == nullptr) {
                list = std::move(node);  // If list is empty, point it to the new node
            } else {
                point_node* iter = list.get();  // Get a raw pointer to traverse the list
                while(iter->next != nullptr) {
                    iter = iter->next.get();  // Use get() to access the raw pointer
                }
                iter->next = std::move(node);  // Append the new node at the end
            }
        }
    }
    
    // print non-zero nodes
    for(point_node* iter = list.get(); iter != nullptr; iter = iter->next.get()) {
        std::cout << "(x, y): " << iter->p.x << ", " << iter->p.y << std::endl;
    }

    // No need to manually delete nodes; unique_ptr automatically deallocates memory
    return 0;
}

