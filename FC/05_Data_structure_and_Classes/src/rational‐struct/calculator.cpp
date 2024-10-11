/*
Developed by: 
    - David Padilla Orenga, NIA: 946874
    - Inacio Pastore Benaim, NIP: 920576
    - Alisson Zapatier Troya, NIA: 717171

Description:
    This program is a simple calculator that works with rational numbers (fractions). 
    The Rational struct provides the functionality to represent, manipulate, and perform
    arithmetic operations on rational numbers. 

    The program continues to ask for input until both rational numbers entered have 0 as
    numerator, which signals the end of the program.

    The program handles the following features:
    - Addition, subtraction, multiplication, and division of rational numbers.
    - Input validation: Denominators cannot be zero, and operations must be valid (+, -, *, /).
    - The program handles edge cases such as division by zero and unrecognized operators.
*/

#include <iostream>
#include "rational.h"
using namespace std;


int main() {
 
    Rational res = rational();
    Rational n1 = rational();
    Rational n2 = rational();
    char calc;

    while (true){
        
        cout << "Enter a calculation" << endl;
        read(cin, n1);
        cin >> calc;
        read(cin, n2);
        
        if (is_zero(n1) && is_zero(n2)) {
            cout << "End" << endl;
            break;
        }

        if (is_infinite(n1) || is_infinite(n2)) {
            cout << "A denominator cannot be 0." << endl;
            continue;
        }

        if (calc == '+') res = add(n1, n2);
        else if (calc == '-') res = sub(n1, n2);
        else if (calc == '*') res = mul(n1, n2);
        else if (calc == '/') {
            if (is_zero(n2)){
                cout << "Error: division by zero" << endl;
                continue; 
            }
            else res = div(n1, n2); 
        }
        else {
            cout << "Error: " << calc << " Operation not recognized" << endl;
            continue;
        }
        cout << "Output: ";
        write(cout,res);
        cout << endl;

    }

   return 0; 
}


