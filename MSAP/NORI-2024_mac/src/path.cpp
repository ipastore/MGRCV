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

	Color3f computeEmitterRadiance(const Ray3f& ray, const Intersection& its) const {
		EmitterQueryRecord emitterRecord(its.p);
		emitterRecord.ref = ray.o;
		emitterRecord.wi = ray.d;
		emitterRecord.n = its.shFrame.n;
		return its.mesh->getEmitter()->eval(emitterRecord); 
	}

	Color3f executePathTracing(const Scene* scene, Sampler* sampler, const Ray3f& ray, Color3f bsdf_ac, bool russRulActivated) const {

		Color3f Lo(0.);

		// ** RAY INTERSECTION ** //
		Intersection its;

		// Checking if the ray intersects with the scene (OUTPUT: Background)
		if (!scene->rayIntersect(ray, its)){
			Lo = scene->getBackground(ray);
			return bsdf_ac * Lo;
		}

		// Checking if the ray intersects with an emitter (OUTPUT: Emitter radiance)
		if (its.mesh->isEmitter()) {
			Lo = computeEmitterRadiance(ray, its);
			return bsdf_ac * Lo;

		} else {

			// ** BSDF SAMPLING ** //

			// We obtain the new direction wo
			Point2f sample = sampler->next2D();
			BSDFQueryRecord bsdfQR(its.toLocal(-ray.d), sample);
			// We obtain the wo and the bsdf value
			Color3f bsdf = its.mesh->getBSDF()->sample(bsdfQR, sample);
			
			// We must check if the BSDF value is valid, if not we'll return black
			if (bsdf.isZero() || bsdf.hasNaN()) {
				return Color3f(0.0f);
			}

			// ** RUSSIAN ROULETTE ** //

			float russin_prob = std::min(bsdf_ac.maxCoeff(), 0.95f);
			bsdf_ac *= bsdf;
			if ((sampler->next1D() > russin_prob) && russRulActivated){
				return Color3f(0.0f);
			}
			
			// ** IF NOT: GENERATE NEW RAY ** //

			// If ray survives we must update the bsdf_ac using the russian roulette probability
			bsdf_ac /= russin_prob;
			Ray3f next_ray(its.p, its.toWorld(bsdfQR.wo));

			// ** RECURSIVE CALL ** //
			Lo = executePathTracing(scene, sampler, next_ray, bsdf_ac, true);
		}

		return Lo;
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

			// ** BSDF SAMPLING ** //

            // We obtain the new direction wo
            BSDFQueryRecord bsdfQR(its.toLocal(-ray.d), sampler->next2D());
            Color3f bsdf = its.mesh->getBSDF()->sample(bsdfQR, sampler->next2D());
            
            // We must check if the BSDF value is valid, if not we'll return black
            if (bsdf.isZero() || bsdf.hasNaN()) {
                return Color3f(0.0f);
            }

			// ** RECURSIVE CALL ** //

            // Generate the new ray
            Ray3f second_ray(its.p, its.toWorld(bsdfQR.wo));
            second_ray.mint = Epsilon;
            second_ray.maxt = std::numeric_limits<float>::infinity();
            
            // Lo = bsdf * Li(scene, sampler, second_ray);
			Lo = executePathTracing(scene, sampler, second_ray, bsdf, true);
        }

		return Lo;
	}

	std::string toString() const {
		return "PathTracing []";
	}
};

NORI_REGISTER_CLASS(PathTracing, "path");
NORI_NAMESPACE_END