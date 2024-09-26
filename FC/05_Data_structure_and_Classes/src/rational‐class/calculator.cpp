#include <iostream>
#include "rational.h"
using namespace std;


int main() {
 
    Rational res = Rational();
    Rational r1 = Rational();
    Rational r2 = Rational();
    char calc;

    while (true){
        
        cout << "Enter a calculation" << endl;
        r1.read(cin);
        cin >> calc;
        r2.read(cin);
        
        if (r1.is_zero() && r2.is_zero()) {
            cout << "End" << endl;
            break;
        }

        if (r1.is_infinite() || r2.is_infinite()) {
            cout << "A denominator cannot be 0." << endl;
            continue;
        }

        if (calc == '+') res = r1.add(r2);
        else if (calc == '-') res = r1.sub(r2);
        else if (calc == '*') res = r1.mul(r2);
        else if (calc == '/') res = r1.div(r2); 
        else {
            cout << "Error: " << calc << " Operation not recognized" << endl;
            continue;
        }

        cout << "Output: ";
        res.write(cout);
        cout << endl;
    }

   return 0; 
}


