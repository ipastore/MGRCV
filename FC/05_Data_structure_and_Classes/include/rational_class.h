#pragma once

#include <iostream>

// Comentario para ver push
class Rational {
private:
   int num;
   int den;
public:
   Rational(int num=0, int den=1);
   void write(std::ostream& os, const Rational& r);
};


// Rational rational(int num = 0, int den = 1);

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

