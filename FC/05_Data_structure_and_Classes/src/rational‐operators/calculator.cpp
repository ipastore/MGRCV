#include <iostream>
#include "rational.h"
using namespace std;

/*
Developed by: 
    - David Padilla Orenga, NIA: 946874
    - Inacio Pastore Benaim, NIP: 920576
    - Alisson Zapatier Troya, NIA: 717171

Description:
    This program is a simple calculator that works with rational numbers (fractions). 
    The Rational class provides the functionality to represent, manipulate, and perform
    arithmetic operations on rational numbers. Input and output operations 
    are supported through the overloaded stream operators (>> and <<).

    The program continues to ask for input until both rational numbers entered have 0 as
    numerator, which signals the end of the program.

    The program handles the following features:
    - Addition, subtraction, multiplication, and division of rational numbers.
    - Input validation: Denominators cannot be zero, and operations must be valid (+, -, *, /).
    - The program handles edge cases such as division by zero and unrecognized operators.
*/

int main() {
 
    Rational res = Rational();
    Rational r1 = Rational();
    Rational r2 = Rational();
    char calc;

    while (true){
        
        cout << "Enter a calculation" << endl;
        cin >> r1 >> calc >> r2;
        
        if (r1.is_zero() && r2.is_zero()) {
            cout << "End" << endl;
            break;
        }

        if (r1.is_infinite() || r2.is_infinite()) {
            cout << "A denominator cannot be 0." << endl;
            continue;
        }

        if (calc == '+') res = r1 + r2;
        else if (calc == '-') res = r1 - r2;
        else if (calc == '*') res = r1 * r2;
        else if (calc == '/') res = r1 / r2; 
        else {
            cout << "Error: " << calc << " Operation not recognized" << endl;
            continue;
        }

        cout << "Output: " << res << endl; 
    }

   return 0; 
}


