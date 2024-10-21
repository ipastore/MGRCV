/*
 * Developed by: 
 *     - David Padilla Orenga, NIA: 946874
 *     - Inacio Pastore Benaim, NIP: 920576
 *     - Alisson Zapatier Troya, NIA: 717171
 * 
 * 1. What is the memory related error? 
 *  There is a double delete for a single pointer. In lines 43 and 45 there is going to be an undefined behaviour.
 * 
 * Now, build and run the application in debug mode with valgrind and answer the following questions.
 * 1. Does valgrind find the problem?
 * Yes: 
 ==30169== Invalid free() / delete / delete[] / realloc()
==30169==    at 0x484B600: operator delete(void*, unsigned long) (in /usr/lib/aarch64-linux-gnu/valgrind/vgpreload_memcheck-arm64-linux.so)
==30169==    by 0x109933: main (in /home/ipastore/MGRCV/FC/08_Memory_leaks/src/build-debug/incorrect_delete)
==30169==  Address 0x4cdcc90 is 12 bytes after a block of size 4 free'd
==30169==    at 0x484B600: operator delete(void*, unsigned long) (in /usr/lib/aarch64-linux-gnu/valgrind/vgpreload_memcheck-arm64-linux.so)
==30169==    by 0x109913: main (in /home/ipastore/MGRCV/FC/08_Memory_leaks/src/build-debug/incorrect_delete)
==30169==  Block was alloc'd at
==30169==    at 0x484A3C4: operator new(unsigned long) (in /usr/lib/aarch64-linux-gnu/valgrind/vgpreload_memcheck-arm64-linux.so)
==30169==    by 0x1098B3: main (in /home/ipastore/MGRCV/FC/08_Memory_leaks/src/build-debug/incorrect_delete)
==30169== 
==30169== 
==30169== HEAP SUMMARY:
==30169==     in use at exit: 0 bytes in 0 blocks
==30169==   total heap usage: 3 allocs, 4 frees, 73,732 bytes allocated
==30169== 
==30169== All heap blocks were freed -- no leaks are possible
==30169== 
==30169== For lists of detected and suppressed errors, rerun with: -s
==30169== ERROR SUMMARY: 1 errors from 1 contexts (suppressed: 0 from 0)
 * 
 * 2. How would you solve it?
 *  Done in fixed_incorrect_delte.cpp
 */
 #include <iostream>

int usage(int argc, char const* argv[]) {
      //read the number of steps from the command line
  if (argc != 2) {
    std::cerr << "Invalid syntax: incorrect_delete <int_value>" << std::endl;
    exit(1);
  }

  int n = std::stoi(argv[1]);
  return n;
}

int main(int argc, char const* argv[])
{

  auto n = usage(argc, argv);

  int* int_ptr = new int(n);

  std::cout << "Current value: " << *int_ptr << std::endl;

  delete int_ptr;

  int_ptr+=4;

  delete int_ptr;
}
