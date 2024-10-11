/*
Developed by: 
    - David Padilla Orenga, NIA: 946874
    - Inacio Pastore Benaim, NIP: 920576
    - Alisson Zapatier Troya, NIA: 717171

Changes:

    - Added Boke brush: applies a Boca Juniors flag.
    - Added Orange and Teal brush: applies an orange and teal color grading.
*/


#include "brushes.h"
#include "color.h"
#include "grayscale.h"
#include "boke.h"
#include "orangeandtealbrush.h"


std::vector<std::unique_ptr<Brush>> brushes() {
    std::vector<std::unique_ptr<Brush>> bs;
    bs.push_back(std::make_unique<Color>("red",255,0,0));
    bs.push_back(std::make_unique<Color>("green",0,255,0));
    bs.push_back(std::make_unique<Color>("blue",0,0,255));
    bs.push_back(std::make_unique<Grayscale>());
    bs.push_back(std::make_unique<Boke>());
    bs.push_back(std::make_unique<OrangeAndTealBrush>());

    return bs;
} 
