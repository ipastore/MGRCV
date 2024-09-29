#include <iostream>
#include <memory>

/*
 * Developed by: 
 *     - David Padilla Orenga, NIA: 946874
 *     - Inacio Pastore Benaim, NIP: 920576
 * 
 * QUESTIONS: 
 * 1. How many objects (variables) are allocated in the stack and in the heap?
 *  STACK OBJECTS:
 *  - There are two objects allocated in the STACK: sp1 and sp2.
 *  - The sp1 object is created in the main scope, so it is accessible for all the main domain. 
 *  - The sp2 object is created in the new scope indicated by the inner {}. It is only reachable in this {} domain.
 *  - Both objects are shared pointers, meaning that they are automatically destroyed when they go out of range.
 *  - Sp1 is destroyed when main function ends. Sp2 object is destroyed when {} domain ends.
 *  
 *  HEAP OBJECTS:
 *  - There is one object allocated on the heap: std::make_shared<int>(1).
 *  - This is a integer pointer with a value of 1 allocated on the heap and managed by sp1 and sp2.
 * 
 * 2. Can both sp1 and sp2 modify the object in the heap?
 *  - Yes, they can. We are using smart pointer that are used to control the ownership of pointers in order to 
 *    read and write on the heap. In this case, we are using shared_pointer which allow multiple pointers to share
 *    ownership of the same resource. This means that both can read from and write to the same object in the heap.
 * 
 * 3. Does the destruction of sp2 releases the memory allocated in the heap?
 *  - No, it doesn't. The shared_pointers for a same address don't realease the memory on the heap until all the shared
 *    pointers pointing to this address are destroyed.
 *  - The output of the code give us:
 *      sp1.use_count(): 1
 *      sp1.use_count(): 2
 *      sp2.use_count(): 2
 *      sp1.use_count(): 1
 *      Is sp1 unique? 1
 *  - As seen in the output, when sp2 goes out of scope, this smart_pointer or reference is destroyed but sp1 refernce is
 *    still remaining unitl main functions ends. Therefore, sp2 destruction doesn't mean the realeases of the memory allocated
 *    on the heap.    
 * 
 */

int main() {
  auto sp1 = std::make_shared<int>(1);
  std::cout << "sp1.use_count(): " << sp1.use_count() << std::endl;
  { // new scope
    std::shared_ptr<int> sp2(sp1);
    std::cout << "sp1.use_count(): " << sp1.use_count() << std::endl;
    std::cout << "sp2.use_count(): " << sp2.use_count() << std::endl;
  }
  std::cout << "sp1.use_count(): " << sp1.use_count() << std::endl;
  std::cout << "Is sp1 unique? " << sp1.unique() << std::endl;
}
