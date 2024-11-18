#include <nori/warp.h>
#include <nori/integrator.h>
#include <nori/scene.h>
#include <nori/emitter.h>
#include <nori/bsdf.h>

NORI_NAMESPACE_BEGIN

class DirectMaterialSampling : public Integrator {
public:
	DirectMaterialSampling(const PropertyList& props) {
		/* No parameters this time */
	}

	Color3f Li(const Scene* scene, Sampler* sampler, const Ray3f& ray) const {
		Color3f Lo(0.);

		// ** FIRST RAY INTERSECTION ** //

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
		}


		// ** SAMPLING THE SECOND RAY ** //
        
		BSDFQueryRecord bsdfQR(its.toLocal(-ray.d), sampler->next2D());
		// We obtain the wo and the bsdf value
		Color3f bsdf = its.mesh->getBSDF()->sample(bsdfQR, sampler->next2D());
		
		// We must check if the BSDF value is valid, if not we'll return black
        if (bsdf.isZero() || bsdf.hasNaN()) {
            return Color3f(0.0f);
        }

		Intersection its2;
		Ray3f second_ray(its.p, its.toWorld(bsdfQR.wo));

		// Checking if the SECOND ray intersects with the scene (OUTPUT: Zero)
		if (!scene->rayIntersect(second_ray, its2)){
			Color3f backgroundColor = scene->getBackground(second_ray);
			return backgroundColor * bsdf;
		}
		
		// Checking if the SECOND ray intersects with an emitter (OUTPUR: Emitter radiance)
		if (its2.mesh->isEmitter()) {	
			EmitterQueryRecord emitterRecord_2(its2.p);
			emitterRecord_2.ref = second_ray.o;
			emitterRecord_2.wi = second_ray.d;
			emitterRecord_2.n = its2.shFrame.n;
			Color3f Le = its2.mesh->getEmitter()->eval(emitterRecord_2); 
			Lo = bsdf * Le;
		}
	

		return Lo;
	}

	std::string toString() const {
		return "Direct Material Sampling []";
	}
};

NORI_REGISTER_CLASS(DirectMaterialSampling, "direct_mats");
NORI_NAMESPACE_END