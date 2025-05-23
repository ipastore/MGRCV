#include <iostream>

/*
Developed by: 
    - David Padilla Orenga, NIA: 946874
    - Inacio Pastore Benaim, NIP: 920576

Please read the variables.cpp program and briefly describe what the program does, specially in
terms of pointer arithmetic.

DESCRIPTION:
    - In terms of functionallity, the programs analyzes an array of integers and determine which are even and which are not.
    - In terms of pointer arithmetic, the pointer called as 'lynn_ptr' pointing to the previous array 'lynn_conway'. The memory
    direction it is pointing corresponds to the first element of this array. This method allows to acces all the elements inside
    the array using pointer arithmetic. Wa have observed several to access the data ways in these code:
        · Inside the loop, lynn_ptr[i] accesses directly the array elements.
        · In the cout message it also accces the same value of the array element but dereferencing the pointer direction with an offset: *(lynn_ptr + i).

Then, what would happen if n is changed to 7 in the loop. ¿Does the compiler gives any error?

OUTPUT:
    4 is even.
    8 is even.
    12 is even.
    2 is even.
    0 is even.

DESCRIPTION:
    - When n = 7 in the loop, the program tries to acces lynn_conway[6], which is out of bounds of the array.
    - No compilation error is given by the compilator. Instead we get a '0 is even'.
    - The sintax allows to do that because even if we are out of the array bondaries, this new memory direction is accesible. When we declare an array,
    the memory for that array is allocated consecutively. If we access an index that is out of bounds, we might still be reading from a nearby memory
    location that hasn't been overwritten or modified. In this case this location contains a 0 which works for the program but maybe, it's possible that
    this memory contains something else, leading the program to some undefined behavoiur.

And, finally, could you remove the pointers and use a range-for loop. The final file should be named
variables_wout_pointers.cpp.

SOLUTION:
    - 
*/

namespace { // anonymous namespace
    const size_t n = 7;
    unsigned int lynn_conway[n] = {3, 4, 7, 8, 12, 2};

    bool is_even(unsigned int a) {
        unsigned int mask = 0x1;
        return (a & mask) == 0;
    }
} // anonymous namespace


int main(){
    unsigned int *lynn_ptr=lynn_conway;
    
    for(size_t i = 0; i < 7; ++i) {
        if(is_even(lynn_ptr[i])) {
            std::cout << *(lynn_ptr+i) << " is even." << std::endl;
        }
    }
}
