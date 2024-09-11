//FALTA: armar tests mas completos

#include <iostream>
#include "functions.h"

using namespace std;

void test_power(float base, int exp, float expected){
    float actual_value = powerWithExpInt(base, exp);
    // if(actual_value != expected){
    if(abs(actual_value - expected) > 0.02){
        cout << "Error: powerWithExpInt(" << base << ", " << "exp) = " << actual_value << endl;
        cout << "Expected " << expected << endl;
    }   
}

void test_factorial (int number, int expected){
    int actual_value = factorial(number);
    if(actual_value != expected){
        cout << "Error: factorial("<< number<<") = " << actual_value << endl;
        cout << "Expected " << expected << endl;
    }
}

void test_arcsin(float x, int iterations, float expected){
    float actual_value = arcsin(x, iterations);
    // if(actual_value != expected){
    if(abs(actual_value - expected) > 0.02){
        cout << "Error: arcsin("<< x <<", "<< iterations << ") = " << actual_value << endl;
        cout << "Expected " << expected << endl;
    }
}

int main() {
    // //Power function tests
    test_power(2,2,4);
    test_power(4,6,4096);
    test_power(4,0,1);
    test_power(3.6,3,46.656);

    //Factorial function tests
    test_factorial(0,1);
    test_factorial(1,1);
    test_factorial(2,2);
    test_factorial(3,6);
    test_factorial(4,24);
   
    //Arcsin function tests
    test_arcsin(3.0/4.0, 1, 42.9718);
    test_arcsin(3.0/4.0,10,48.59);
    test_arcsin(3.0/4.0,20,48.59);
    test_arcsin(3.0/4.0,30,48.59);
    test_arcsin(3.0/4.0,50,48.59);

    return 0;   
}