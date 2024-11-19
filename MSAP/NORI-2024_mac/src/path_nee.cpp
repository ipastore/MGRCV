#include <nori/warp.h>
#include <nori/integrator.h>
#include <nori/scene.h>
#include <nori/emitter.h>
#include <nori/bsdf.h>

NORI_NAMESPACE_BEGIN

class PathTracingNEE : public Integrator {
public:
	PathTracingNEE(const PropertyList& props) {
		/* No parameters this time */
	}
 
    Color3f emiter_sampler(const Scene* scene, Sampler* sampler, const Ray3f& ray, const Intersection& its) const {
        Color3f Lo(0.);

        // Randomly choose an emitter
        float pdfEmitter;	// this is the probability density of choosing an emitter
		const Emitter* em = scene->sampleEmitter(sampler->next1D(), pdfEmitter);

        // Sample a point on the emitter and get its radiance
		EmitterQueryRecord emitterRecord(its.p);
        Color3f Lem = em->sample(emitterRecord, sampler->next2D(), 0.f);

        // Create a shadow ray to see if the point is in shadow
        Ray3f shadowRay(its.p, emitterRecord.wi);
        shadowRay.maxt = (emitterRecord.p - its.p).norm();

        // Check if the ray intersects with an emitter or if the first intersection in shadows
        Intersection shadow_its;
        bool inShadow = scene->rayIntersect(shadowRay, shadow_its);

        if (!inShadow || shadow_its.t >= (emitterRecord.dist - Epsilon)){
            BSDFQueryRecord bsdfQR_ls(its.toLocal(-ray.d), its.toLocal(emitterRecord.wi), its.uv, ESolidAngle);
			Color3f bsdf = its.mesh->getBSDF()->eval(bsdfQR_ls);
            float denominator = pdfEmitter * emitterRecord.pdf;
            if (denominator > Epsilon){	// to avoid division by 0 (resulting in NaNs and anoying warnings)
                emitterRecord.dist = its.t;
                Lo = (Lem * its.shFrame.n.dot(emitterRecord.wi) * bsdf) / denominator;
			}
		}
		return Lo;
	}
   

	Color3f Li(const Scene* scene, Sampler* sampler, const Ray3f& ray) const {
		Color3f Lo(0.);
        Color3f Le_em(0.);

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

            if (bsdfQR.measure == EDiscrete)
            {
                Lo = bsdf * Li(scene, sampler, second_ray);
            }else{
                Le_em = emiter_sampler(scene, sampler, second_ray, its);
                Lo = bsdf * (Le_em + Li(scene, sampler, second_ray));
            }
            
        }

		return Lo;
	}

	std::string toString() const {
		return "PathTracingNEE []";
	}
};

NORI_REGISTER_CLASS(PathTracingNEE, "path_nee");
NORI_NAMESPACE_END