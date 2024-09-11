#include <iostream>
using namespace std;


/*
Main function containing a basic calculator.
n1 and n2 are floats 
calc is the character containing the operator
res is the output result
We have seen that:
    - a + b : Breaks the cin buffer.
    - ´ (tilde en español): Breaks the cin buffer.
    - Using a comma instead of a point for decimmals: breaks the cin buffer.
*/
int main() {
    float n1, n2, res; 
    char calc;

    while (true){
        
        cout << "Enter a calculation" << endl;
        cin >> n1 >> calc >> n2;
        
        if (0==n1 && 0==n2) {
            cout << "End" << endl;
            break;
        }

        if (calc == '+') res = n1 + n2;
        else if (calc == '-') res = n1 - n2;
        else if (calc == '*') res = n1 * n2;
        else if (calc == '/') {
            if (0==n2){
                cout << "Error: division by zero" << endl;
                continue; 
            }
            else res = n1 / n2; 
        }
        else {
            cout << "Error: " << calc << " Operation not recognized" << endl;
            continue;
        }
        cout << "Output: " << res << endl; 
    }

   return 0; 
}
