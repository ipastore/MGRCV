/*
 * Developed by: 
 *     - David Padilla Orenga, NIA: 946874
 *     - Inacio Pastore Benaim, NIP: 920576
 *     - Alisson Zapatier Troya, NIP: 717171
 * 
 * */

#include <iostream>
#include "functions.h"

double exponential (double base, int exponente){
    double res = 1; 
    for (int i=1; i<=abs(exponente) ; i++){
        res*=base;
    }
    
    if (exponente>=0)  return res;
    else        return (1/res);    
}

int factorial(int number) {
    int factorial=1;
   
    while(number>1) factorial*=(number--);
    
    return factorial;
}

double arcsin(double x, int iterations){
    double result = 0; 

    for (int n=0; n<iterations; n++){
        double numerator = factorial(2*n) * exponential(x,(2*n+1));
        double denominator = exponential(4,n) * exponential(factorial(n),2) * (2*n+1);
        result += numerator/denominator;
    }

    result = result * 180.0/3.14159;
    return result;
}
