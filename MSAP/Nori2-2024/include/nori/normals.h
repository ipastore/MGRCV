#include <nori/integrator.h>

NORI_NAMESPACE_BEGIN

class NormalIntegrator : public Integrator {
public:
    // Constructor: initializes the class with properties from the scene file
    NormalIntegrator(const PropertyList &props);

    // Compute the radiance value for a given ray. Just return green here
    Color3f Li(const Scene *scene, Sampler *sampler, const Ray3f &ray) const;

    // Return a human-readable description for debugging purposes
    std::string toString() const;

protected:
    std::string m_myProperty;
};

NORI_NAMESPACE_END
