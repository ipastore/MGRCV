#include <nori/integrator.h>

NORI_NAMESPACE_BEGIN

class NormalIntegrator : public Integrator {
public:
    // Constructor: initializes the class with properties from the scene file
    NormalIntegrator(const PropertyList &props) {
        m_myProperty = props.getString("myProperty");
        std::cout << "Parameter value was: " << m_myProperty << std::endl;
    }

    // Compute the radiance value for a given ray. Just return green here
    Color3f Li(const Scene *scene, Sampler *sampler, const Ray3f &ray) const {
        return Color3f(0, 1, 0); // Return green color for every intersection
    }

    // Return a human-readable description for debugging purposes
    std::string toString() const  {
        return tfm::format(
            "NormalIntegrator[\n"
            "  myProperty = \"%s\"\n"
            "]",
            m_myProperty
        );
    }

protected:
    std::string m_myProperty;
};

// Register the class to the Nori system
NORI_REGISTER_CLASS(NormalIntegrator,"normals");
NORI_NAMESPACE_END