/*
 * Developed by: 
 *     - David Padilla Orenga, NIA: 946874
 *     - Inacio Pastore Benaim, NIP: 920576
 *     - Alisson Zapatier Troya, NIP: 717171
 * 
 * Main function containing a a program to calculate the degree of a set of rectangular triangules using 
 * a Taylor series for the approximation for the arcsin function. 
 *     - c input correponds to the cathetus value.
 *     - h input correponds to the hypotenuse value.
 *     - The output returns the angle value in degrees.
 * 
 * We have seen that:
 *     - With 30 iterations it breaks.
 *     - Also with values of 9/10 and above it loses precision.
 */

#include <iostream>
#include "functions.h"
using namespace std;


int main() {

    double c, h, result;

    while (true){

        // Get Cathetus and check impossible triangle
        cout << "Enter the cathetus c of the triangle" << endl;
        cin >> c;
        if (c<=0){
            cout<<"Impossible triangle. C must be positive"<<endl;
            cout<<"END"<<endl; 
            break;
        }
        
        // Get Hypotenuse and check impossible triangle
        cout << "Enter the hypotenuse h of the triangle"<< endl;
        cin >> h;
            if (h<=0){
            cout<<"Impossible triangle. H must be positive"<<endl;
            cout<<"END"<<endl; 
            break;
        }

        // Check cathetus/hypothenuse ratio
        if (c/h>=1){
            cout<<"Impossible triangle. The cathetus cannot be greater than the hypotenuse."<<endl;
            cout<<"END"<<endl; 
            break;
        } 
    
        result = arcsin(c/h, 10);

        cout << "The angle is " << result << " degrees." <<endl;
    }

  return 0;   
}
