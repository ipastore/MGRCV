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

        // get cateto ARMAR FUNC
        cout << "Introudce el cateto del  triángulo" << endl;
        cin >> c;
        if (c<=0){
            cout<<"Has introducido un cateto prohibido"<<endl;
            cout<<"END"<<endl; 
            break;
        }
        // get hipotenusa ARMAR FUNC
        cout << "Introduce la hipotenusa h del triángulo" << endl;
        cin >> h;
            if (h<=0){
            cout<<"Has introducido un hipotenusa prohibido"<<endl;
            cout<<"END"<<endl; 
            break;
        }

        if (c/h>=1){
            cout<<"Has introducido un triangulo rectangulo prohibido"<<endl;
            cout<<"END"<<endl; 
            break;
        } 
    
        result = arcsin(c/h, 40);

        cout << "El angulo es de " << result << "°." <<endl;
    }

  return 0;   
}


