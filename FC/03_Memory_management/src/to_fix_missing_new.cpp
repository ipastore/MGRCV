#include <iostream>
using namespace std;

/*
Developed by: 
    - David Padilla Orenga, NIA: 946874
    - Inacio Pastore Benaim, NIP: 920576

ERROR WITHOUT THE FIX:
    array[0]: 6
    [1]    73522 segmentation fault 

SOLUTION:
    - Segmentation fault means that the code is trying to access a forbidden memory address. We have comment this line to avoid that.
    In general, pointing to a random memory direction should be avoided.

    - We also have added to the delete_two_array funtion '[]' to the delete in order to delete the entire array. Using only 'delete'
    is undefined behaviour and it may appear to work but there is no guarantee. Take into account that the last
    cout printing "array[1]" was also commented out because the delete[] was performed before and there is no reason
    to access it again.
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
    array[0] = 6;
    array[1] = 7;
    cout << "array[0]: " << array[0] << endl;
    cout << "array[1]: " << array[1] << endl;
    delete_two_array(array);
    // array=reinterpret_cast<int*>(0xDEADBEEF); // Is 0xDEADBEEF a valid address?
    // cout << "array[1]: " << array[1] << endl;
}
