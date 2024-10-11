#pragma once
#include "brush.h"

class OrangeAndTealBrush : public Brush {
public:
    std::string name() const override;
    void edit(unsigned char& r, unsigned char& g, unsigned char& b) override;
};