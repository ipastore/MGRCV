#include <iostream>
using namespace std;

int main() {
    int n, fn;

    cout << "Introduce a number"<<endl;
    cin >> n;

// While Opcion 1 
    // fn = n-1;
    
    // if(n < 0 ){
    //     cout << "No se puede hacer factorial de un numero negativo" << endl;
    // } else if (n==0){
    //     cout << "El factorial de 0 es 1" << endl;
    // } else {   
    //      while (fn>1){
    //         n = n * (fn);
    //         fn--;
    //     }
    //     cout << "The factorial is " << n << endl;
    // }

    
// While Opcion 2
    fn=1;
    while(n>1) fn*=(n--); 
    cout << "The factorial is " << fn << endl;

// // For Opcion 1
//     fn=1;
//     for (;n>1;){
//          fn*=(n--); 
//     }
//     cout << "The factorial is " << fn << endl;


    

    return 0;
}