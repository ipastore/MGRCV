/*
Developed by: 
    - David Padilla Orenga, NIA: 946874
    - Inacio Pastore Benaim, NIP: 920576
    - Alisson Zapatier Troya, NIP: 717171

ERROR GIVEN WITHOUT THE FIX:
    to_fix_double_delete(72441,0x1fb817ac0) malloc: Double free of object 0x13de05fc0
    to_fix_double_delete(72441,0x1fb817ac0) malloc: *** set a breakpoint in malloc_error_break to debug
    [1]    72441 abort  

SOLUTION:
    One of the deletes (either the function or the delete) must be removed since both are trying to free
    the same memory direction.
*/

#include <iostream>
using namespace std;


int* create_int_ptr(){
    int* ptr = new int;
    return ptr;
}

void delete_int_ptr(int* ptr){
    delete ptr;
}

int main(){
    int* int_ptr = create_int_ptr();
    delete_int_ptr(int_ptr);
    // delete int_ptr;
}

