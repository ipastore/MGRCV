/**
 * Developed by: 
    - David Padilla Orenga, NIA: 946874
    - Inacio Pastore Benaim, NIP: 920576
    - Alisson Zapatier Troya, NIA: 717171

 * Rational.h
 * Definition of the Rational class to represent and manipulate rational numbers.
 * 
 * The Rational class provides functionality for representing rational numbers (fractions),
 * and performing arithmetic operations, comparison, and input/output values using the terminal.
 * The class ensures that rational numbers are stored in their reduced form and provides methods
 * for common arithmetic operations like addition, subtraction, multiplication, and division.
 * 
 * Key Features:
 * - Arithmetic operations: add, sub, mul, div.
 * - Zero and infinity checks: is_zero, is_infinite.
 * - Input/Output: Overloaded stream operators.
 * - Getter for both, numrator and denominator values.
 * - Input and output operations are supported through the overloaded stream operators (>> and <<) which don't belong to the class.
 **/

#pragma once
#include <iostream>


class Rational {
   private:
      int num;
      int den;
   public:
      
      // Constructor
      Rational(int num=0, int den=1);

      // Getters
      int get_num() const;
      int get_den() const;

      // Getters
      void set_num(int n);
      void set_den(int d);

      // Operaciones aritmeticas
      Rational operator+(const Rational& other) const;
      Rational operator-(const Rational& other) const;
      Rational operator*(const Rational& other) const;
      Rational operator/(const Rational& other) const;

      // Zero & infinity check
      bool is_zero() const;
      bool is_infinite() const;

   private:
      // Auxiliar methods, private
      int gcf(int a, int b);

};

// Input/Output
std::ostream& operator<<(std::ostream& os, const Rational& r);
std::istream& operator>>(std::istream& is, Rational& r);
