fmatrix async_col_matrix_multiply(const fmatrix &a, const fmatrix &b, const index)
{

    
    fmatrix c{rows, cols};

    for(size_t i = 0; i < a.rows(); ++i) {
       float val{0.0f};
       for(size_t k = 0; k < a.cols(); ++k) {
           val+=a(i, k) * b(k, index);
       }
       c(i, index)=val;
    }
    return c;
}

int main () 
{
    fmatrix matrixA;
    fmatrix matrixB;
    int num_threads = 0;
   const size_t num_threads_possible = std:🧵:hardware_concurrency();
   if (num_threads_possible == 1)
    {
        num_threads = 1;
    }
   else
    {
        num_threads = matrixA.cols();
    }

    std::vector<std::future<fmatrix>> futures;

    for (size_t i = 0; i < num_threads; ++i)
    {
        futures.push_back(std::async(std::launch::async, async_col_matrix_multiply, std::cref(matrixA), std::cref(matrixB), i));
    }

    fmatrix result{rows, cols};

    for (size_t i = 0; i < num_threads; ++i)
    {
        fmatrix result_ = futures[i].get();
        if (num_threads == 1)
        {
            result = result_;
        }
        else
        {
            for (size_t j = 0; j < result_.rows(); ++j)
            {
                result(j, i) = result_(j, i);
            }
            // ALTERNATIVELY
            result.hstack(result_); // Assuming matrix <float> class has a hstack method defined
        }
        // Combine the results
    }


}