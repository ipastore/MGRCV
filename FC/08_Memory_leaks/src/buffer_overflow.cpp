/*
 * Developed by: 
 *     - David Padilla Orenga, NIA: 946874
 *     - Inacio Pastore Benaim, NIP: 920576
 *     - Alisson Zapatier Troya, NIA: 717171
 * 
 *  Output: 
 * 
 * ipastore@linuxserver:~/MGRCV/FC/08_Memory_leaks/src/build-debug$ ./buffer_overflow 
1
2
3
4
=================================================================
==32100==ERROR: AddressSanitizer: stack-buffer-overflow on address 0xffffc58ee370 at pc 0xaaaac87f7808 bp 0xffffc58ee2d0 sp 0xffffc58ee2f0
READ of size 4 at 0xffffc58ee370 thread T0
    #0 0xaaaac87f7804 in main /home/ipastore/MGRCV/FC/08_Memory_leaks/src/buffer_overflow.cpp:7
    #1 0xffffaa120e0c in __libc_start_main ../csu/libc-start.c:308
    #2 0xaaaac86f2410  (/home/ipastore/MGRCV/FC/08_Memory_leaks/src/build-debug/buffer_overflow+0x10410)

Address 0xffffc58ee370 is located in stack of thread T0 at offset 48 in frame
    #0 0xaaaac87f75b0 in main /home/ipastore/MGRCV/FC/08_Memory_leaks/src/buffer_overflow.cpp:3

  This frame has 1 object(s):
    [32, 48) 'a' (line 4) <== Memory access at offset 48 overflows this variable
HINT: this may be a false positive if your program uses some custom stack unwind mechanism, swapcontext or vfork
      (longjmp and C++ exceptions *are* supported)
SUMMARY: AddressSanitizer: stack-buffer-overflow /home/ipastore/MGRCV/FC/08_Memory_leaks/src/buffer_overflow.cpp:7 in main
Shadow bytes around the buggy address:
  0x200ff8b1dc10: 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
  0x200ff8b1dc20: 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
  0x200ff8b1dc30: 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
  0x200ff8b1dc40: 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
  0x200ff8b1dc50: 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
=>0x200ff8b1dc60: 00 00 00 00 00 00 00 00 f1 f1 f1 f1 00 00[f3]f3
  0x200ff8b1dc70: 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
  0x200ff8b1dc80: 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
  0x200ff8b1dc90: 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
  0x200ff8b1dca0: 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
  0x200ff8b1dcb0: 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
Shadow byte legend (one shadow byte represents 8 application bytes):
  Addressable:           00
  Partially addressable: 01 02 03 04 05 06 07 
  Heap left redzone:       fa
  Freed heap region:       fd
  Stack left redzone:      f1
  Stack mid redzone:       f2
  Stack right redzone:     f3
  Stack after return:      f5
  Stack use after scope:   f8
  Global redzone:          f9
  Global init order:       f6
  Poisoned by user:        f7
  Container overflow:      fc
  Array cookie:            ac
  Intra object redzone:    bb
  ASan internal:           fe
  Left alloca redzone:     ca
  Right alloca redzone:    cb
  Shadow gap:              cc
==32100==ABORTING
 */

#include <iostream>

int main() {
  int a[4] = {1, 2, 3, 4};

  for(size_t i = 0; i < 32; ++i) {
    std::cout << a[i] << std::endl;
  }
}
