// FALTA catchear errore de iteraciones negativas en arcsin
// Catchear number menor a 0 en factorial
// Se podria meter comentarios por linea

#include <iostream>
#include "functions.h"

float powerWithExpInt (float base, int exponente){
    float res = 1; 
    for (int i=1; i<=abs(exponente) ; i++){
        res*=base;
    }
    
    if (exponente>=0)  return res;
    else        return (1/res);    
}

// No catcheamos si number es menor a 0
int factorial(int number) {
    int factorial=1;
   
    while(number>1) factorial*=(number--);
    
    return factorial;
}


// Func arcsin. Catchaer errores de iteraiorns <= 0 y number
float arcsin(float x, int iterations){
    float result = 0; 

    for (int n=0; n<iterations; n++){
        float numerator = factorial(2*n) * powerWithExpInt(x,(2*n+1));
        float denominator = powerWithExpInt(4,n) * powerWithExpInt(factorial(n),2) * (2*n+1);
        result += numerator/denominator;
    }

    result = result * 180.0/3.14159;
    return result;
}