/**
 * Developed by: 
    - David Padilla Orenga, NIA: 946874
    - Inacio Pastore Benaim, NIP: 920576
    - Alisson Zapatier Troya, NIA: 717171

 * Rational.h
 * Definition of the Rational Struct to represent and manipulate rational numbers.
 * 
 * The Rational Struct provides functionality for representing rational numbers (fractions),
 * and performing arithmetic operations, comparison, and input/output values using the terminal.
 * The Struct ensures that rational numbers are stored in their reduced form and provides methods
 * for common arithmetic operations like addition, subtraction, multiplication, and division.
 * 
 * Key Features:
 * - Arithmetic operations: add, sub, mul, div.
 * - Zero and infinity checks: is_zero, is_infinite.
 **/

#pragma once

#include <iostream>

struct Rational {
   int num, den;
};


Rational rational(int num = 0, int den = 1);

// Input/output

void write(std::ostream& os, const Rational& r);
void read(std::istream& is,Rational& r);

// Operaciones aritmeticas

Rational add(const Rational& r1, const Rational& r2);
Rational sub(const Rational& r1, const Rational& r2);
Rational mul(const Rational& r1, const Rational& r2);
Rational div(const Rational& r1, const Rational& r2);

// Zero check
bool is_zero(const Rational& r);
bool is_infinite(const Rational& r);

