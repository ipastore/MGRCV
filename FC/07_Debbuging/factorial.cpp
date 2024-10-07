#include <iostream>

unsigned int factorial(unsigned int n)
{
  if(n == 0) {
    return 1;
  }

  return factorial(n-1) * n;
}

int main()
{
    std::cout << "Type the n value for computing factorial: ";

    unsigned int n{};
    std::cin >> n;

    auto result = factorial(n);
    std::cout << "factorial(" << n << ") = " << result << std::endl;

    return 0;
}

/*
 * Developed by: 
 *     - David Padilla Orenga, NIA: 946874
 *     - Inacio Pastore Benaim, NIP: 920576
 * 
 * Comments added at the end to do not alterate the number lines for the given explanations.
 *   
 * ### INITIAL DEBBUGING ###
 * 
 * ## Factorial ##
 * 
 * 
 * 1. Why there is no output of the list command? Could you compile the code in a diffent built type to get more information?
 * 
 *    The list command didn't show output because the code was compiled in Release mode, which removes debugging information for optimization.
 *    It is possible to recompile the code in Debug mode in order to retain this information, allowing GDB to access the source code and display it.
 *    Once done this, the list command in GDB displays the source code (10 rows) around the current execution point or a specified function, helping
 *    understand the program.
 * 
 * 
 * ## Factorial with debug information ##
 * 
 * 1. What is the current output of the list command?
 *    
 *    The list command in GDB displays the source code of the program you're debugging, typically showing 10 lines of code at a time (by default)
 *    
 *        (gdb) list
 *        8	
 *        9	  return factorial(n-1) * n;
 *        10	}
 *        11	
 *        12	int main()
 *        13	{
 *        14	    std::cout << "Type the n value for computing factorial: ";
 *        15	
 *        16	    unsigned int n{};
 *        17	    std::cin >> n;
 *    
 *    In this case, since we stopped at breakpoint in main (row 13) we are able to see part of the factorial function declared before main and part
 *    of the main content.
 * 
 * 
 * 2. Create a breakpoint for the factorial function an then run the program until the breakpoint? What is the backtrace at this
 * point of the execution?
 * 
 *    The backtrace shows the call stack, revealing the sequence of function calls leading to the current point in execution.
 *    
 *        (gdb) backtrace
 *        #0  factorial (n=3)
 *            at /home/padidavid/Documents/unizar/master/repositorios/MGRCV/FC/07_Debbuging/factorial.cpp:5
 *        #1  0x000055555555528a in main ()
 *            at /home/padidavid/Documents/unizar/master/repositorios/MGRCV/FC/07_Debbuging/factorial.cpp:19
 *    
 *    Frame #0: The current function is factorial(n=3), stopped at line 5 where n == 0 is checked. we are here since we created a breakpoint in
 *    factorial (line 5).
 *    Frame #1: factorial(3) was called from main() at line 19.
 *    At /home/padidavid/Documents/.../factorial.cpp:13 shows the exact location in the source file where the program is currently paused. In this
 *    case, it's line 13 of factorial.cpp, which is the beginning of the main() function.
 *    
 *    This helps visualize how the recursion unfolds and where the program is currently paused. Moreover, since factorial is recursive, each frame
 *    will represents a different call in the recursion, so the backtrace will be bigger in each iteration.
 * 
 * 
 * 3. Please delete the breakpoint for the factorial function and create another breakpoint in the line 9 of factorial.cpp source
 * file and run the program again. Please include the used commands in your response.
 * 
 *    (gdb) delete 2 ----------------> Delete the existing breakpoint for factorial
 *    (gdb) break factorial.cpp:9 ---> Set a new breakpoint at line 9
 *    (gdb) run ---------------------> Run the program again
 * 
 * 
 * ## CHANGING THE VALUES OF VARIABLES ##
 * 
 *    What we are doing here is stpping at line 5 breakpoint where we have just enter the factorial function and we are
 *    about to check if n (value of factorial) is 0. When using (p &n), it retrieves the memory address of the variable n.
 *    Using this address and the comand (set *((unsigned int *) <address>) = <value>), we modify the value of the variable 
 *    n by directly modifying the memory at its address. TThis change changes the behavior of this value from that point forward.
 *    
 *    After modifying the variable, we use p n to print and confirm that the value of n has been updated. Finally we uses cont
 *    and finished to finish the program and the value returned is 1 no matter the input we gave it at the beginning because we
 *    have change the n value while debugging using set.
 *    
 *    Moreover if we use Layout Split, GDB provides the option to splits the screen to view the assembly code alongside the source
 *    code using the layout split command. It helps understand how high-level C++ code translates to low-level machine instructions.
 *    After inspecting the assembly code, we can return to the original view using tui disable to exit TUI mode.
 * 
 * 
 * ## CONDITIONAL BREAKPOINTS ##
 * 
 *    Conditional breakpoints are USED when WE want to stop the program when certain conditions are met. This allows us to
 *    focus on specific cases without manually checking for those conditions. In this case if we put a n bigger than 5 it
 *    won't stop in the facttorial recursive loop until n reaches a value minor than 5. For example for 7 we've seen, using
 *    the backtrace command, that the factorial functions has already been called 3 times before stop at the breakpoint. 
 *    
 *        (gdb) break factorial  if n < 5
 *        Breakpoint 1 at 0x1218: file /home/padidavid/Documents/unizar/master/repositorios/MGRCV/FC/07_Debbuging/factorial.cpp, line 5.
 *        (gdb) run
 *        Starting program: /home/padidavid/Documents/unizar/master/repositorios/MGRCV/FC/07_Debbuging/build-debbuging/factorial 
 *        Type the n value for computing factorial: 7
 *        
 *        Breakpoint 1, factorial (n=4)
 *            at /home/padidavid/Documents/unizar/master/repositorios/MGRCV/FC/07_Debbuging/factorial.cpp:5
 *        warning: Source file is more recent than executable.
 *        5	  if(n == 0) {
 *        (gdb) backtrace
 *        #0  factorial (n=4)
 *            at /home/padidavid/Documents/unizar/master/repositorios/MGRCV/FC/07_Debbuging/factorial.cpp:5
 *        #1  0x0000555555555232 in factorial (n=5)
 *            at /home/padidavid/Documents/unizar/master/repositorios/MGRCV/FC/07_Debbuging/factorial.cpp:9
 *        #2  0x0000555555555232 in factorial (n=6)
 *            at /home/padidavid/Documents/unizar/master/repositorios/MGRCV/FC/07_Debbuging/factorial.cpp:9
 *        #3  0x0000555555555232 in factorial (n=7)
 *            at /home/padidavid/Documents/unizar/master/repositorios/MGRCV/FC/07_Debbuging/factorial.cpp:9
 *        #4  0x000055555555528a in main ()
 *            at /home/padidavid/Documents/unizar/master/repositorios/MGRCV/FC/07_Debbuging/factorial.cpp:19
 *        (gdb) 
 * 
 */

