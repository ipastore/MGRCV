/*
 * Developed by: 
 *     - David Padilla Orenga, NIA: 946874
 *     - Inacio Pastore Benaim, NIP: 920576
 *     - Alisson Zapatier Troya, NIA: 717171
 * 
 */

#include <cassert>
#include <iostream>
#include <memory>

int
usage(int argc, const char* argv[]) {
  //read the number of steps from the command line
  if (argc != 2) {
    std::cerr << "Invalid syntax: leak_factorial <int_value>" << std::endl;
    exit(1);
  }

  int n = std::stoi(argv[1]);
  return n;
}

int leak_factorial(int n)
{
    auto result = std::unique_ptr<int>{nullptr};

    assert(n >= 0);

    result = std::make_unique<int>(1);

    if(n == 0) {
      return *result;
    }

    auto next_val = std::make_unique<int>(leak_factorial(n-1) * n);

    return *next_val;
 }

int main(int argc, const char **argv)
{

  auto n = usage(argc, argv);

  auto result = leak_factorial(n);
  std::cout << "Factorial of " << n << " equals " << result << std::endl;

  return 0;
}
