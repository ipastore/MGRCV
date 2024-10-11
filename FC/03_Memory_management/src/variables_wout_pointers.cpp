/*
 * Developed by: 
 *     - David Padilla Orenga, NIA: 946874
 *     - Inacio Pastore Benaim, NIP: 920576
 *     - Alisson Zapatier Troya, NIA: 717171
 * 
 * SOLUTION:
 *     - Simplified Access: Each element is accessed as value in the loop, making the code more readable and concise.
 *     - Instead of using a pointer to access the array, the code now directly iterates over lynn_conway using a range-based for loop. So we would
 *      never try to acces an space of memory outside the boundaries of the array.
 *     - Moreover, an error like n=7 initializating only six of the sevent elements won't be a problem because of the zero-initialitation of all the
 *      elements of the array. In this case, we can be sure that even if we just put six elements of a seven elemts array, the last one would be zero. 
 *     - This is a good example of how, sometimes, avoiding pointers is safer.
 */

#include <iostream>


namespace {
    const size_t n = 7;
    unsigned int lynn_conway[n] = {3, 4, 7, 8, 12, 2}; // Here we have a 0 in lynn_conway[6]. There is no unpredictible behaviour

    bool is_even(unsigned int a) {
        unsigned int mask = 0x1;
        return (a & mask) == 0;
    }
}

int main() {
    for (const unsigned int value : lynn_conway) { // Code iterates ONLY over the size of the array.
        if (is_even(value)) {
            std::cout << value << " is even." << std::endl;
        }
    }
    return 0;
}
