#include <iostream>
using namespace std;

/*
 * Developed by: 
 *     - David Padilla Orenga, NIA: 946874
 *     - Inacio Pastore Benaim, NIP: 920576
 *     - Alisson Zapatier Troya, NIP: 717171
 * 
 * ERROR WITHOUT THE FIX:
 *     array[0]: 6
 *     [1]    73522 segmentation fault 
 * 
 * SOLUTION:
 *     - Segmentation fault means that the code is trying to access a forbidden memory address. To fix this we implemented
 *     a new pointer with a valid address. Afterwards, to demonstrate the functionality of reassigning a pointer, first we delete
 *     former pointer, then reassigned it to the valid address.
 * 
 *     SOLUTION:
 *     - Segmentation fault means that the code is trying to access a forbidden memory address. We have comment this line to avoid that.
 *     In general, pointing to a random memory direction should be avoided because we would have an undefined behaviour.
 * 
 *     - We also have added to the delete_two_array funtion '[]' to the delete in order to free the entire array memory. Using only 'delete'
 *     is undefined behaviour and it may appear to work but there is no guarantee.
 */



int* create_two_array(){
    int* ptr = new int[2];
    return ptr;
}

void delete_two_array(int* ptr){
    delete[] ptr;
}

int main(){
    int* array=create_two_array();
    int* valid_address = create_two_array();

    array[0] = 6;
    array[1] = 7;
    valid_address[0] = 34;
    valid_address[1] = 12;
    cout << "array[0]: " << array[0] << endl; // It prints 6
    cout << "array[1]: " << array[1] << endl; // It prints 7
    delete_two_array(array);
    // array=reinterpret_cast<int*>(0xDEADBEEF); // Not recommended to try to acces to a random memory adress. Undefined behaviour.
    array = valid_address; // We've played in oreder to reassing the 'array' pointer to an existing one.
    cout << "array[0]: " << array[0] << endl; // It prints 34 since is now poiting to the same adress as 'valid_address'.
    cout << "array[1]: " << array[1] << endl; // It prints 12 since is now poiting to the same adress as 'valid_address'.
    delete_two_array(valid_address); // This line deallocate the memory where  'array' and 'valid_address' are pointing. Next two lines would have an undifined behaviour.
    cout << "array[0]: " << array[0] << endl; // Random number --> UNDEFINED BEHAVIOUR 
    cout << "array[1]: " << array[1] << endl; // Random number --> UNDEFINED BEHAVIOUR 
}

