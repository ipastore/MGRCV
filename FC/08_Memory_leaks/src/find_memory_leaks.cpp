#include <cassert>
#include <iostream>
#include <memory>

std::unique_ptr<int[]> allocate_many(size_t n)
{
	assert((n>0) && "Cannot allocate zero-length array");

	// return new int[n];
  return std::make_unique<int[]>(n);
}

void allocate_many_ref(std::unique_ptr<int[]>& int_ptr_ref, size_t n)
{
	assert((n>0) && "Cannot allocate zero-length array");

	int_ptr_ref = std::make_unique<int[]>(n);

	// consider exceptions as well
	// https://learn.microsoft.com/en-us/cpp/cpp/errors-and-exception-handling-modern-cpp?view=msvc-170
	assert((int_ptr_ref != nullptr) && "Returned invalid pointer");
}

int main()
{
  size_t n;
  std::cout << "Please type the number of elements: " << std::endl;
  std::cin >> n;

  // int* int_ptr =allocate_many(n);
  auto int_ptr = allocate_many(n);

  std::unique_ptr<int[]> int_ptr_2;
  allocate_many_ref(int_ptr_2, n);

  // int* int_ptr_2;
  // allocate_many_ref(int_ptr_2, n);

  // delete[] int_ptr;
  // delete[] int_ptr_2;

  return 0;
}
