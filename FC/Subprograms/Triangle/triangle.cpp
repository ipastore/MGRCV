// Falta preguntar por armar funcion de power con exponente float
// Preguntar por precision 


#include <iostream>
#include "functions.h"
using namespace std;

int main() {

    float c, h, result;
    // Con 30 iteraciones se rompe.
    // Ademas con valores de 9/10 para arriba pierde precision
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
    
        result = arcsin(c/h, 40);

        cout << "The angle is " << result << "degrees." <<endl;
    }

  return 0;   
}


