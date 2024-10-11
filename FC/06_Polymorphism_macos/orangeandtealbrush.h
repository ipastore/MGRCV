/*
Developed by: 
    - David Padilla Orenga, NIA: 946874
    - Inacio Pastore Benaim, NIP: 920576
    - Alisson Zapatier Troya, NIA: 717171

*/

#pragma once
#include "brush.h"

class OrangeAndTealBrush : public Brush {
public:
    std::string name() const override;
    void edit(unsigned char& r, unsigned char& g, unsigned char& b) override;
};