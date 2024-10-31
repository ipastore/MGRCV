#include <nori/integrator.h>
#include <nori/scene.h>

NORI_NAMESPACE_BEGIN

class DepthIntegrator : public Integrator {
    public:
        DepthIntegrator(const PropertyList &props) {
            /* No parameters this time */
        }


    Color3f Li(const Scene *scene, Sampler *sampler, const Ray3f &ray) const override {

        // Find the surface that is visible in the requested direction
        Intersection its;
        if (!scene->rayIntersect(ray, its))
            return scene->getBackground(ray);
        
        // Return the depth for each intersected pixel.
        
        float distance = its.t;

        if (distance == 0){
            return Color3f(1.0f);
        }
        
        return Color3f(1.0f / distance);


    }

    std::string toString() const override {
        return "DepthIntegrator[]";
    }

};

NORI_REGISTER_CLASS(DepthIntegrator, "depth");

NORI_NAMESPACE_END
