/*
Developed by: 
    - David Padilla Orenga, NIA: 946874
    - Inacio Pastore Benaim, NIP: 920576
    - Alisson Zapatier Troya, NIA: 717171
*/

#pragma once
#include "brush.h"

class Boke : public Brush {
private:
    static int counter; // Static variable to keep track of the number of calls to the edit method

public:
    std::string name() const override;
    void edit(unsigned char& r, unsigned char& g, unsigned char& b) override;
    virtual ~Boke() = default;

}; 

