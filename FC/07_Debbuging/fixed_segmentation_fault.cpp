#include <iostream>
#include <limits>
#include <random>

namespace {

  // float farray[128]={};
  std::vector<float> farray(127, 0.0f);

  template <typename T>
    size_t find_max_pos(const std::vector<T>& array)
    {
      T max{std::numeric_limits<T>::min()};

      size_t pos{0};

      for(size_t i = 0; i < array.size(); ++i)
      {
        if(array.at(i) > max) {
          pos = i;
          max=array.at(i);
        }
      }

      return pos;
    }

  template <typename T>
    void fill_array(std::vector<T>& array)
    {
      std::random_device rd;
      std::mt19937 gen(rd());
      std::uniform_real_distribution<float> dis {0.0f, 1.0f};

      for(size_t i = 0; i < array.size(); ++i)
      {
        array.at(i)=dis(gen);
      }
    }
}

int main()
{

  fill_array(farray);

  auto pos = find_max_pos(farray);

  std::cout << "The maximum value of farray is: " << farray[pos] << std::endl;

  return 0;
}

