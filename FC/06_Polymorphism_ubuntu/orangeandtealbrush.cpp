#include "orangeandtealbrush.h"
#include <algorithm>

std::string OrangeAndTealBrush::name() const {
    return "orange_and_teal";
}

bool isOrange(const unsigned char& r, const unsigned char& g, const unsigned char& b) {
    return (r > 180 && g >= 90 && b < 80);
}

bool isTeal(const unsigned char& r, const unsigned char& g, const unsigned char& b) {
    return (b > 180 && g >= 90 && r < 80);
}


void OrangeAndTealBrush::edit(unsigned char& r, unsigned char& g, unsigned char& b) {

    unsigned char luminance = static_cast<unsigned char>(0.299 * r + 0.587 * g + 0.114 * b);

    // If the pixel is already in the orange or teal range, skip modifying it
    if (isOrange(r, g, b) || isTeal(r, g, b)) {
        return;
    }

    // Apply the orange or teal tint based on luminance and thresholds
    if (luminance < 64) {
        // Blacks -> Deeper teal
        r = std::max(static_cast<int>(r) - 40, 0);
        g = std::min(static_cast<int>(g) + 10, 255);
        b = std::min(static_cast<int>(b) + 40, 255);
    } else if (luminance >= 64 && luminance < 128) {
        // Darks -> Lighter teal
        r = std::max(static_cast<int>(r) - 20, 0);
        g = std::min(static_cast<int>(g) + 10, 255);
        b = std::min(static_cast<int>(b) + 20, 255);
    } else if (luminance >= 128 && luminance < 192) {
        // Lights -> Lighter orange
        r = std::min(static_cast<int>(r) + 20, 255);
        g = std::min(static_cast<int>(g) + 5, 255);
        b = std::max(static_cast<int>(b) - 20, 0);
    } else {
        // Whites -> Deeper orange
        r = std::min(static_cast<int>(r) + 40, 255);
        g = std::min(static_cast<int>(g) + 10, 255);
        b = std::max(static_cast<int>(b) - 40, 0);
    }
}