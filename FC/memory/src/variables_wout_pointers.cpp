#include <iostream>

#include <iostream>

/*
Developed by: 
    - David Padilla Orenga, NIA: 946874
    - Inacio Pastore Benaim, NIP: 920576

And, finally, could you remove the pointers and use a range-for loop. The final file should be named
variables_wout_pointers.cpp.

SOLUTION:
    - Instead of using a pointer to access the array, the code now directly iterates over lynn_conway using a range-based for loop.
    - Simplified Access: Each element is accessed as value in the loop, making the code more readable and concise.
    - In this case, an error like n=7 (out of range of the array) cannot happend because the loop is based on the lynn_conway lenght.
    - This is a good example of how, sometimes, avoiding pointers is safer.
*/

namespace { // anonymous namespace
    const size_t n = 6;
    unsigned int lynn_conway[n] = {3, 4, 7, 8, 12, 2};

    bool is_even(unsigned int a) {
        unsigned int mask = 0x1;
        return (a & mask) == 0;
    }
} // anonymous namespace

int main() {
    for (unsigned int value : lynn_conway) { 
        if (is_even(value)) {
            std::cout << value << " is even." << std::endl;
        }
    }
    return 0;
}
