#include "rational.h"


// Initialize
Rational::Rational(int num, int den) {
	int m = gcf(num,den);
	this->num = num/m;
	this->den = den/m; 
}

// Getters
int Rational::get_num() const{
	return this->num;
}

int Rational::get_den() const{
	return this->den;
}

// Setter
void Rational::set_num(int n){
 	this->num = n;
}

void Rational::set_den(int d){
	this->den = d;
}

// Auxiliar functions, private
int Rational::gcf(int a, int b) {
	return ( b==0 ? a : gcf(b,a%b) );
}

// Operations
Rational Rational::operator+(const Rational& other) const {
	return Rational(
		this->num * other.get_den() + other.get_num() * this->den,
		this->den * other.get_den()
	);

}

Rational Rational::operator-(const Rational& other) const {
	return Rational(
		this->num * other.get_den() - other.get_num() * this->den,
		this->den * other.get_den()
	);
}

Rational Rational::operator*(const Rational& other) const {
	return Rational(
		this->num * other.get_num(),
		this->den * other.get_den()
	);
}

Rational Rational::operator/(const Rational& other) const {
	return Rational(
		this->num * other.get_den(),
		this->den * other.get_num()
	);
}

// Zero check
bool Rational::is_zero() const{
	return this->num == 0;
}

// Infinity check
bool Rational::is_infinite() const {
	return this->den == 0;
}

// Output
std::ostream& operator<<(std::ostream& os, const Rational& r) {
   os << r.get_num() << "/" << r.get_den();
   return os;
} 


// Input
std::istream& operator>>(std::istream& is, Rational& r) {
   int numerator, denominator; char dummy;
   is >> numerator >> dummy >> denominator;
	r.set_num(numerator); r.set_den(denominator);
   return is;
}
