# Nori2

Some troubleshooting for building:
## 1. Compilation problems in Linux

If you are using the GCC compiler (g++) version 13 (you can check with the command "g++ -v"), chances are the code will not compile because of compatiblity issues. There are several possible solutions

### Solution 1: downgrade to g++12 (easy)

You can downgrade to g++-12:

```sudo apt install g++-12```

And then tell cmake that it should use g++-12 instead:

```cmake -DCMAKE_CXX-COMPILER=/usr/bin/g++-12 ..```

You can find where g++-12 is located with

```whereis g++-12```

### Solution 2: use clang instead (easy)

You can use clang instead of g++, which is usually slightly faster at compiling. Check the [Mitsuba docs](https://mitsuba.readthedocs.io/en/latest/src/developer_guide/compiling.html#linux) for installing and using clang to compile nori through cmake. 

Install the libraries:

```sudo apt install clang-10 libc++-10-dev libc++abi-10-dev cmake ninja-build```


Set clang-10 as the compiler:

```export CC=clang-10 export CXX=clang++-10```

Create the build files, and compile:
```
mkdir build
cd build
cmake -GNinja ..
ninja
```

From now on, you only have to be in the build directory and run ```ninja``` to recompile.


### Solution 3: Fix the g++13 issue (maybe harder)
You have to modify one line in one file of the _tbb_ library. See [here](https://discourse.mc-stan.org/t/cmdstan-installation-fails-on-linux-issue-with-tbb/31125/2) for more details.

Credits: Hugo Mateo for finding the fix


## 2. Compiling in Windows without cmake-gui

First, you need cmake and the Visual Studio compiler (_not_ Visual Studio Code) so that you can generate the build files:

```
cmake -G "Visual Studio 17 2022" -A x64 -B build
```

Then, to compile:
```
cmake --build build --config Release -j 10
```

(you can use ```-j 10``` to compile with 10 threads, etc.)

You can sometimes use --config Debug to enable debug mode. Be careful with this, the renders will be ~10 times slower.

___________________________
If you have any more issues contact your course instructors and we'll be happy to help :)

