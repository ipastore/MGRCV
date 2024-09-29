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



std::unique_ptr<point_node> copy_point(const point& p)
{
    auto point_ptr = std::make_unique<point_node>(p);
    if (p.x > 1)
        return nullptr;
    return point_ptr;
}


int main(){
    const size_t n = 4;
    point points[n]={ {0, 0}, {1, 1}, {0, 0}, {1, 1} };

    // save points whose coordinates are not (0, 0)
    std::unique_ptr<point_node> list = nullptr;

    for (const auto& p : points) {
        // point_node* node = new point_node(p, nullptr);
        std::unique_ptr<point_node> node  = std::make_unique<point_node>(p);

        if (p.not_zero()) {
            if (list == nullptr) {
                list = std::move(node);;
            } else { 
                point_node* current = list.get(); 
                while (current->next != nullptr) {
                    current = current->next.get();
                }
                current->next = std::move(node);
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

