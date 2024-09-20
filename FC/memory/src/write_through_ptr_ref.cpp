#include <iostream>
using namespace std;

/*
Developed by: 
    - David Padilla Orenga, NIA: 946874
    - Inacio Pastore Benaim, NIP: 920576

Main program to modify a number entered by the user with pointers and references
*/


int main() {

    int number;

    cout << "Enter a number" << endl;
    cin >> number;

    int* const increment_ptr = &number;
    *increment_ptr += 2;
    cout << "Number after incrementation with Constant Pointer: " << number << endl;

    int& ref = number;
    ref -= 2;
    cout << "Number after substraction using reference: " << number << endl;

}

