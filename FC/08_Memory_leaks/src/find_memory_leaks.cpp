/*
 * Developed by: 
 *     - David Padilla Orenga, NIA: 946874
 *     - Inacio Pastore Benaim, NIP: 920576
 *     - Alisson Zapatier Troya, NIA: 717171
 * 
 * 1. In which lines there are memory leaks? Please concisely explain the issues, if any. 
 *    There are two sources of memory leaks:
 *    a. In the function `allocate_many`, where memory is allocated with `new int[n]`
 *       but never freed.
 *    b. In the function `allocate_many_ref`, where memory is allocated to the 
 *       reference pointer but also not freed, leading to a memory leak.
 * 
 * 2. Solve the memory leaks with smart pointers in the new correct_factorial.cpp file, that your submission should include. 
 *    Done in correct_find_memory_leaks.cpp
 * 
 * 3. Verify with Valgrind that correct_factorial does not have any memory leak. 
 *     Done in correct_find_memory_leaks.cpp: "==5107== All heap blocks were freed -- no leaks are possible"
 * 
 */
 
#include <cassert>
#include <iostream>

int* allocate_many(size_t n)
{
	assert((n>0) && "Cannot allocate zero-length array");

	return new int[n];
}

void allocate_many_ref(int*& int_ptr_ref, size_t n)
{
	assert((n>0) && "Cannot allocate zero-length array");

	int_ptr_ref = new int[n];

	// consider exceptions as well
	// https://learn.microsoft.com/en-us/cpp/cpp/errors-and-exception-handling-modern-cpp?view=msvc-170
	assert((int_ptr_ref != nullptr) && "Returned invalid pointer");
}

int main()
{
  size_t n;
  std::cout << "Please type the number of elements: " << std::endl;
  std::cin >> n;

  allocate_many(n);


  int* int_ptr;
  allocate_many_ref(int_ptr, n);

  return 0;
}