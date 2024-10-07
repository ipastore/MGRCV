#include <iostream>
#include <limits>
#include <random>

namespace {

  float farray[128]={};

  template <typename T>
    size_t find_max_pos(const T* array, const size_t n)
    {
      T max{std::numeric_limits<T>::min()};

      size_t pos{0};

      for(size_t i = 0; i < (n<<8); ++i)
      {
        if(array[i] > max) {
          pos = i;
          max=array[i];
        }
      }

      return pos;
    }

  template <typename T>
    void fill_array(T* array, const size_t n)
    {
      std::random_device rd;
      std::mt19937 gen(rd());
      std::uniform_real_distribution<float> dis {0.0f, 1.0f};

      for(size_t i = 0; i < n; ++i)
      {
        array[i]=dis(gen);
      }
    }
}

int main()
{

  fill_array(farray, 256);

  auto pos = find_max_pos(farray, 256);

  std::cout << "The maximum value of farray is: " << farray[pos] << std::endl;

  return 0;
}


/*
 * Developed by: 
 *     - David Padilla Orenga, NIA: 946874
 *     - Inacio Pastore Benaim, NIP: 920576
 * 
 * Comments added at the end to do not alterate the number lines for the given explanations.
 *   
 * ### DEBBUGING ###
 * 
 * 1. Please describe how the segmentation faults occur.
 *
 *    The problem here causing the segmentation fault is an Out-of-Bounds access problem due to surpassing and access the array
 *    dimensions. The array `farray` is declared with a size of 128 elements (float farray[128] = {}) but the code attempts to
 *    fill and process 256 elements in `fill_array(farray, 256)` overwriting some memory addresses we should not have access to.
 *    Moreover, `find_max_pos(farray, 256)`.tries to do exactly the same but even worse. In this case, the loop iterates over a
 *    large number of elements due to the left shift operation (`n << 8`), which increases the number of iterations. To be more 
 *    accurate, n << 8 means shifting the binary representation of n 8 bits to the left, which is equivalent to multiplying by 2^8
 *    (256). This causes the program to access memory locations outside the allocated array, resulting in a segmentation fault. 
 * 
 * 
 * 2. Please comment how to find the exacts errors with gdb.
 * 
 *    (gdb) break main
 *    (gdb) break 35 if i > 128 -----------------------> To see what happens when i exceeds the array size
 *    (gdb) run
 *         Breakpoint 1, main ()
 *    (gdb) p farray
 *         $1 = {0 <repeats 128 times>} ---------------> The array is well initialized
 *    (gdb) cont
 *         Breakpoint 2, (anonymous namespace)::fill_array<float> (
 *             array=0x55555555a180 <(anonymous namespace)::farray>, n=256)
 *         36	        array[i]=dis(gen);
 *    (gdb) p i
 *         $2 = 129
 *    (gdb) p farray[i]
 *         $3 = 0 --------------------------------------> We shouldn't be modifiying the array at this point
 *    (gdb) p farray
 *         (Give us 128 values of the farray fullfilled with values differents from zero)
 *    (gdb) cont
 *    (gdb) cont
 *         Breakpoint 2,...
 *    (gdb) p i
 *         $2 = 130
 *    (gdb) p farray[i]
 *       $6 = 0
 *    (gdb) p farray
 *         (Give us the exact same 128 array without no changes)
 *    (gdb) p farray[i-1]
 *       $7 = 0.336515456 ---------------------------------------------> It doen't belong to the array we have just printed. We are writting somewhere else.
 *    (gdb) delete 2 --------------------------------------------------> Delete the breakpoint to continue to the next function pending to analyse.
 *    (gdb) break 18 if i>250 -----------------------------------------> To see what happens when i exceeds the array size in the max_pos function
 *    (gdb) cont
 *       Breakpoint 3, (anonymous namespace)::find_max_pos<float> ....
 *       18	        if(array[i] > max) {
 *    (gdb) p i 
 *       $8 = 251 -----------------------------------------------------> i variable surpass the array size without problems
 *    (gdb) p n<<8
 *       $9 = 65536 ---------------------------------------------------> The statment for exit the loop is enormous compared to the array size. This is going to cause a segmentation fault at some point.
 *    
 * 
 * 3. Would an std::vector container fix the issue?
 * 
 *    It is true that std::vector is a dynamic container that can automatically grow or shrink as needed. When using std::vector,
 *    we don’t need to manually allocate a fixed size like float farray[128], we can wait until knowing the array dimension. It can
 *    be declared without specifying the size upfront and then resize it dynamically using vector::resize().
 *    
 *    If we still access elements using the [] operator, bounds checking is not done. However, std::vector provide a safe access methods
 *    like at() which perform bounds checking amd if we try to access an out-of-bounds index, it will throw an exception (std::out_of_range)
 *    instead of causing a segmentation fault. This gives us a safer way to access elements.Moreover, this doesn't solve the acces memory 
 *    provoqued by (`n << 8`). This must be corrected. 
 * 
 * 
 */

