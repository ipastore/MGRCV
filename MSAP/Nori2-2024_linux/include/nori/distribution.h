

#pragma once

#include <nori/common.h>
#include <nori/object.h>

NORI_NAMESPACE_BEGIN


struct Distribution1D{
public:

    Distribution1D(const float* f, int n);
    float SampleContinuous(float u, float* pdf, int* off) const;
    float DiscretePdf(int index) const;
    int Count() const;
    std::vector<float> cdf, func;
    float funcInt;


};

struct Distribution2D{
public:
    Distribution2D(const float* func, int nu, int nv);
    Point2f SampleContinuous2(const Point2f& u, float* pdf) const;
    float DiscretePdf2(const Point2f& p) const;

private:
    std::vector<std::unique_ptr<Distribution1D>> pConditionalV;
    std::unique_ptr<Distribution1D> pMarginal;
};

NORI_NAMESPACE_END
