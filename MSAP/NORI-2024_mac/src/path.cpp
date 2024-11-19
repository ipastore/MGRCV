#include <nori/warp.h>
#include <nori/integrator.h>
#include <nori/scene.h>
#include <nori/emitter.h>
#include <nori/bsdf.h>

NORI_NAMESPACE_BEGIN

class PathTracing : public Integrator {
public:
	PathTracing(const PropertyList& props) {
		/* No parameters this time */
	}
 

	Color3f Li(const Scene* scene, Sampler* sampler, const Ray3f& ray) const {
		Color3f Lo(0.);

		// ** RAY INTERSECTION ** //

		Intersection its;

		// Checking if the ray intersects with the scene (OUTPUT: Background)
		if (!scene->rayIntersect(ray, its))
			return scene->getBackground(ray);

		// Checking if the ray intersects with an emitter (OUTPUR: Emitter radiance)
		if (its.mesh->isEmitter()) {
			EmitterQueryRecord emitterRecord(its.p);
			emitterRecord.ref = ray.o;
			emitterRecord.wi = ray.d;
			emitterRecord.n = its.shFrame.n;
			return its.mesh->getEmitter()->eval(emitterRecord); 
		} else {

            // We obtain the new direction wo
            BSDFQueryRecord bsdfQR(its.toLocal(-ray.d), sampler->next2D());
            // We obtain the wo and the bsdf value
            Color3f bsdf = its.mesh->getBSDF()->sample(bsdfQR, sampler->next2D());
            
            // We must check if the BSDF value is valid, if not we'll return black
            if (bsdf.isZero() || bsdf.hasNaN()) {
                return Color3f(0.0f);
            }

            // Russian Roulette termination
            float q = std::min(bsdf.maxCoeff(), 0.99f);
            if (sampler->next1D() > q)
                return Color3f(0.0f);

            // Generate the new ray
            Ray3f second_ray(its.p, its.toWorld(bsdfQR.wo));
            second_ray.mint = Epsilon;
            second_ray.maxt = std::numeric_limits<float>::infinity();
            

            Lo = bsdf * Li(scene, sampler, second_ray);
        }

		return Lo;
	}

	std::string toString() const {
		return "PathTracing []";
	}
};

NORI_REGISTER_CLASS(PathTracing, "path");
NORI_NAMESPACE_END