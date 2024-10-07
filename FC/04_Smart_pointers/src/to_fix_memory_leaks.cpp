#include <iostream>
#include <memory>

/*
 * Developed by: 
 *     - David Padilla Orenga, NIA: 946874
 *     - Inacio Pastore Benaim, NIP: 920576
 *     - Alisson Zapatier Troya, NIA: 717171
 * 
 * Starting from the max_min_heap_2d_matrix.cpp code, replace all dynamic memory operations
 * with   smart   pointers.   The   resulting   file   should   be   named   max_min_sp_2d_matrix.cpp.   The 
 * maximum and minimum values should be printed by reading the value from an smart pointer as well.
 * 
 * EXPLANATION:
 *   
 *  */

struct point
{
    int x, y;
    bool not_zero() const { return x!=0 || y!=0; };
};


struct point_node
{
    point p;
    std::unique_ptr<point_node> next; 
    point_node(const point& other_p) : p(other_p), next(nullptr) {}
};


std::unique_ptr<point_node> copy_point(const point& p)
{
    auto point_ptr = std::make_unique<point_node>(p);
    if (p.x > 1)
        return nullptr;
    return point_ptr;
}


void assign_next_node(std::unique_ptr<point_node>& current_node, std::unique_ptr<point_node>& new_node){
    if (current_node->next == nullptr){
        current_node->next = std::move(new_node);
    }
    else {
        assign_next_node(current_node->next, new_node);
    }
}


void print_list_nodes(std::unique_ptr<point_node>& list)
{
    if (list->next == nullptr)
    {
        std::cout << "(x, y): " << list->p.x << ", " << list->p.y << std::endl;
    } else {
        std::cout << "(x, y): " << list->p.x << ", " << list->p.y << std::endl;
        print_list_nodes(list->next);
    }
}


int main(){
    const size_t n = 7;
    point points[n]={ {0, 0}, {1, 1}, {0, 0}, {8, 5}, {0, 0}, {0, 0}, {3, 4} };

    // List to save points whose coordinates are not (0, 0). List will contain the first point and the pointer to the second one
    std::unique_ptr<point_node> list = nullptr;

    for (const auto& p : points) {

        std::unique_ptr<point_node> node  = std::make_unique<point_node>(p);

        if (p.not_zero()) {
            if (list == nullptr) {
                list = std::move(node);;
            } else { 
                assign_next_node(list, node);
            }
        }
    }
    
    // print non-zero nodes
    print_list_nodes(list);

    // No need to manually delete nodes; unique_ptr automatically deallocates memory
    return 0;
}

