#include <nori/integrator.h>
#include <nori/scene.h>

NORI_NAMESPACE_BEGIN

class NormalIntegrator : public Integrator {
public:
    NormalIntegrator(const PropertyList &props) {
        // Retrieve a property named "myProperty" from the property list
        // m_myProperty = props.getString("myProperty");
        // std::cout << "Parameter value was: " << m_myProperty << std::endl;
    }

    /// Compute the radiance value for a given ray. Just return green here
    Color3f Li(const Scene *scene, Sampler *sampler, const Ray3f &ray) const override {
        // Find the surface that is visible in the requested direction
        Intersection its;
        if (!scene->rayIntersect(ray, its)) {
            // If no intersection is found, return black (0.0f)
            return Color3f(0.0f);
        }

        // Return the component-wise absolute value of the shading normal as a color
        Normal3f n = its.shFrame.n.cwiseAbs();
        return Color3f(n.x(), n.y(), n.z());
    }

    /// Return a human-readable description for debugging purposes
    std::string toString() const override {
        return "NormalIntegrator[]";
    }

};

NORI_REGISTER_CLASS(NormalIntegrator, "normals");

NORI_NAMESPACE_END
