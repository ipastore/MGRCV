#include <nori/integrator.h>
#include <iostream>
#include <nori/common.h> // Assuming tfm is in common.h

NORI_NAMESPACE_BEGIN

class NormalIntegrator : public Integrator {
    public:
        NormalIntegrator(const PropertyList &props);
        /// Compute the radiance value for a given ray. Just return green here
        Color3f Li(const Scene *scene, Sampler *sampler, const Ray3f &ray) const override;

        /// Return a human-readable description for debugging purposes
        std::string toString() const override;

    protected:
        std::string m_myProperty;
};

NORI_NAMESPACE_END
