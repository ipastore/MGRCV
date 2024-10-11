/*
 * Developed by: 
 *     - David Padilla Orenga, NIA: 946874
 *     - Inacio Pastore Benaim, NIP: 920576
 *     - Alisson Zapatier Troya, NIP: 717171
 * 
 * */

#pragma once

#include <iostream>

/**
 * The function exponential calculates the result of raising a base to a given integer exponent.
 *
 * base represents the base of the exponentiation.
 * exp represents the exponent, which is an integer.
 * The function returns the result of base raised to the power of exp as a double.
 */
double exponential(double base, int exp);

/**
 * The function factorial calculates the factorial of a given non-negative integer.
 *
 * number represents the non-negative integer whose factorial is to be calculated.
 * The function returns the factorial of the number as an integer.
 */
int factorial(int number);

/**
 * The function arcsin calculates the arcsine (inverse sine) of a value using a Taylor series approximation.
 *
 * x represents the input value for which the arcsin is to be calculated.
 * iterations represents the number of terms to consider in the Taylor series for the approximation.
 * The function returns the approximate value of arcsin(x) as a double.
 */
double arcsin(double x, int iterations);
