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

    Color3f computeEmitterRadiance(const Ray3f& ray, const Intersection& its) const {
		EmitterQueryRecord emitterRecord(its.p);
		emitterRecord.ref = ray.o;
		emitterRecord.wi = ray.d;
		emitterRecord.n = its.shFrame.n;
		return its.mesh->getEmitter()->eval(emitterRecord); 
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
            float denominator = pdfEmitter * emitterRecord.pdf;
            if (denominator > Epsilon){
                BSDFQueryRecord bsdfQR_ls(its.toLocal(-ray.d), its.toLocal(emitterRecord.wi), its.uv, ESolidAngle);
                Color3f bsdf = its.mesh->getBSDF()->eval(bsdfQR_ls);
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
		if (!scene->rayIntersect(ray, its)){
			return scene->getBackground(ray);
        }

		// Checking if the ray intersects with an emitter (OUTPUR: Emitter radiance)
		if (its.mesh->isEmitter()) {
			Lo = computeEmitterRadiance(ray, its);
		} else {

            // ** BSDF SAMPLING ** //

            // We obtain the new direction wo
            Point2f sample = sampler->next2D();
            BSDFQueryRecord bsdfQR(its.toLocal(-ray.d), sample);
            Color3f bsdf = its.mesh->getBSDF()->sample(bsdfQR, sample);
            bool sampleLightsFlag = (bsdfQR.measure != EDiscrete);
            float w_mats = sampleLightsFlag ? 0.5f : 1.0f;
            float w_lights = sampleLightsFlag ? 0.5f : 0.0f;
            
            // We must check if the BSDF value is valid, if not we'll return black
            if (bsdf.isZero() || bsdf.hasNaN()) {
                return Color3f(0.0f);
            }

			// ** GENERATE NEW RAY ** //

            // Generate the new ray
            Intersection nex_its;
            Ray3f next_ray(its.p, its.toWorld(bsdfQR.wo));
            next_ray.mint = Epsilon;
            next_ray.maxt = std::numeric_limits<float>::infinity();

            // ** RECURSIVE CALL DEPENDING ON MATERIAL** //

            if(!scene->rayIntersect(next_ray, nex_its)){
                Le_em = emiter_sampler(scene, sampler, ray, its);
                Lo = bsdf * (w_lights * Le_em + w_mats* scene->getBackground(next_ray));
                return Lo;
            }else if(sampleLightsFlag || nex_its.mesh->isEmitter()){
                w_mats = 1.0f;
                Lo = w_mats * bsdf * executePathTracingNEE(scene, sampler, nex_its, next_ray, bsdf, false);
            }else{
                Le_em = emiter_sampler(scene, sampler, ray, its);
                Lo = bsdf * (w_lights * Le_em + w_mats* executePathTracingNEE(scene, sampler, nex_its, next_ray, bsdf, false));
            }
            
        }

		return Lo;
	}

    Color3f executePathTracingNEE(const Scene* scene, Sampler* sampler, Intersection& its, const Ray3f& ray, Color3f bsdf_ac, bool russRulActivated) const {
		Color3f Lo(0.);
        Color3f Le_em(0.);

		// ** RAY INTERSECTION ** //

		// Checking if the ray intersects with the scene (OUTPUT: Background)
		if (!scene->rayIntersect(ray, its)){
            Lo = scene->getBackground(ray);
			return Lo;
        }

		// Checking if the ray intersects with an emitter (OUTPUR: Emitter radiance)
		if (its.mesh->isEmitter()) {
			Lo = computeEmitterRadiance(ray, its);
			return Lo;
		} else {

            // ** BSDF SAMPLING ** //

            // We obtain the new direction wo
            Point2f sample = sampler->next2D();
            BSDFQueryRecord bsdfQR(its.toLocal(-ray.d), sample);
            Color3f bsdf = its.mesh->getBSDF()->sample(bsdfQR, sample);
            bool sampleLightsFlag = (bsdfQR.measure != EDiscrete);
            float w_mats = sampleLightsFlag ? 0.5f : 1.0f;
            float w_lights = sampleLightsFlag ? 0.5f : 0.0f;
            
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
                
            bsdf_ac /= russin_prob;
            

			// ** IF NOT: GENERATE NEW RAY ** //

            // Generate the new ray
            Intersection nex_its;
            Ray3f next_ray(its.p, its.toWorld(bsdfQR.wo));
            next_ray.mint = Epsilon;
            next_ray.maxt = std::numeric_limits<float>::infinity();


            // ** RECURSIVE CALL DEPENDING ON MATERIAL** //

            if(!scene->rayIntersect(next_ray, nex_its)){
                Le_em = emiter_sampler(scene, sampler, ray, its);
                Lo = bsdf * (w_lights * Le_em + w_mats* scene->getBackground(next_ray));
                return Lo;
            }else if(sampleLightsFlag || nex_its.mesh->isEmitter()){
                w_mats = 1.0f;
                Lo = w_mats * bsdf_ac * executePathTracingNEE(scene, sampler, nex_its, next_ray, bsdf, true);
            }else{
                Le_em = emiter_sampler(scene, sampler, ray, its);
                Lo = bsdf_ac * (w_lights * Le_em + w_mats* executePathTracingNEE(scene, sampler, nex_its, next_ray, bsdf, true));
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