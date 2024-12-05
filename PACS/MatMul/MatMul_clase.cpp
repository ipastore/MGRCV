/* 
a  b  c      1 2  3
d  e  f      4 5  6 
g  h  i      7 8  9


a1 + b4 + c7      a2 + b5 + c8      a3 + b6 + c9 
d2 + e5 + f8      d2 + e5 + f8      d3 + e6 + f9 
g3 + h6 + i9      g3 + h6 + i9      g3 + h6 + i9

workitems is Ha * Wb

Matrix A; // Ha * Wa
Matrix B; // Hb * Wb

The for is going to cover the direction of row in matrix A and th direction ofcol in matrix B


global_dim[2]0{#rows,#cols}
clEnqueNDRangeKernel(....,2,&glogbal_dim,...)


// In the non-naive approach

global_dim[2]0{#rows,#cols}
clEnqueNDRangeKernel(....,2,&glogbal_dim,...)
The same code as before

TSxTS = preferred_workgroup_size_multiple

If you want to use vector code: prefferred vectorz sizes
*/