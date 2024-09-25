#include <iostream>
using namespace std;

/*
Developed by: 
    - David Padilla Orenga, NIA: 946874
    - Inacio Pastore Benaim, NIP: 920576

// In which section each variable will be stored?
    - int global_var would be stored in Data.
    - int local_varwould be stored in Stack.
    - int* global_ptr would be stored in Stack, points to Data.
    - int* local_ptr  would be stored in Stack, points to Stack.
*/

int global_var = 1;
 
int main() {

    int local_var = 2;
    int* global_ptr = &global_var;
    int* local_ptr = &local_var;

    cout << "global_var stores: " << global_var << endl;
    cout << "loca_var stores: " << local_var << endl;
    cout << "global_ptr: " << global_ptr << endl;
    cout << "local_ptr: " << local_ptr << endl;

    return 0;
}
