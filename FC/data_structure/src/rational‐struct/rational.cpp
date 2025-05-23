#include "rational.h"

// Auxiliar functions, private
int gcf(int a, int b) {
	return ( b==0 ? a : gcf(b,a%b) );
}

// Initialize
Rational rational(int num, int den) {
	Rational r; 
	int m = gcf(num,den);
	r.num=num/m; r.den=den/m; return r;
}

// Input/output
void write(std::ostream& os, const Rational& r) {
	os<<r.num<<"/"<<r.den;
}

void read(std::istream& is, Rational& r) {
	int num, den; char dummy;
	is>>num>>dummy>>den;
	r = rational(num,den);
}

// Operations
Rational add(const Rational& r1, const Rational& r2) {
	return rational(
		r1.num*r2.den + r2.num*r1.den,
		r1.den*r2.den
	);
}

Rational sub(const Rational& r1, const Rational& r2) {
	return rational(
		r1.num*r2.den - r2.num*r1.den,
		r1.den*r2.den
	);
}

Rational mul(const Rational& r1, const Rational& r2) {
	return rational(
		r1.num*r2.num,
		r1.den*r2.den
	);
}

Rational div(const Rational& r1, const Rational& r2) {
	return rational(
		r1.num*r2.den,
		r1.den*r2.num
	);
}

//Zero check
bool is_zero(const Rational& r) {
	return r.num == 0;
}

bool is_infinite(const Rational& r) {
	return r.den == 0;
}
