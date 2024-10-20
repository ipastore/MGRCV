/*
 * Developed by: 
 *     - David Padilla Orenga, NIA: 946874
 *     - Inacio Pastore Benaim, NIP: 920576
 *     - Alisson Zapatier Troya, NIA: 717171
 * 
 * 1. In which lines there are memory leaks? Please concisely explain the issues, if any. 
 *  In this program there are two types of erros: 
 *    a) The return in line 46 should be "return *next_val;"
 *    b) After initializing the pointers in lines 39 and 45, they are never deleted. Moreover, as the function is recursive, there is a cascade 
 *    of memory leaks. 
 * 
 * 2. Solve the memory leaks with smart pointers in the new correct_factorial.cpp file, that your submission should include. 
 *    Done in correct_factorial.cpp
 * 
 * 3. Verify with Valgrind that correct_factorial does not have any memory leak. 
 *    Done, no memory leaks: "==3363== All heap blocks were freed -- no leaks are possible"
 * 
 * 4. Please repeat the previous 3 questions for the find_memory_leaks.cpp source file. 
 *    Done in find_memory_leaks.cpp
 */
  

#include <cassert>
#include <iostream>

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
    int* result {nullptr};

    assert(n >= 0);

    result = new int(1);

    if(n == 0) {
      return *result;
    }

    int* next_val = new int(leak_factorial(n-1) * n);

    return *result + *next_val;
 }

int main(int argc, const char **argv)
{

  auto n = usage(argc, argv);

  auto result = leak_factorial(n);
  std::cout << "Factorial of " << n << " equals " << result << std::endl;

  return 0;
}

