#include <nori/warp.h>
#include <nori/integrator.h>
#include <nori/scene.h>
#include <nori/emitter.h>
#include <nori/bsdf.h>

NORI_NAMESPACE_BEGIN

class DephtIntegrator : public Integrator {
public:
    DephtIntegrator(const PropertyList& props) {
        /* No parameters this time */
    }

    Color3f Li(const Scene* scene, Sampler* sampler, const Ray3f& ray) const {
        Color3f Lo(0.);

        // Find the surface that is visible in the requested direction
        Intersection its;
        if (!scene->rayIntersect(ray, its))
            // If no intersection is found, return black (0.0f)
            return Color3f(0.0f);

        float distance = (ray.o - its.p).norm();

        if (distance == 0)
        {
            return Color3f(1.0f);
        }
        
        return Color3f(1.0f / distance);
    }

    std::string toString() const {
        return "Direct Whitted Integrator []";
    }
};

NORI_REGISTER_CLASS(DephtIntegrator, "depht_integrator");

NORI_NAMESPACE_END
