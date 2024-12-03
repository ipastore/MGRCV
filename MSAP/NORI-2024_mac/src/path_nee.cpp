#include <nori/warp.h>
#include <nori/integrator.h>
#include <nori/scene.h>
#include <nori/emitter.h>
#include <nori/bsdf.h>

NORI_NAMESPACE_BEGIN

class PathTracingNee : public Integrator {
public:
	PathTracingNee(const PropertyList& props) {
		/* No parameters this time */
	}

    Color3f computeEmitterRadiance(const Ray3f& ray, const Intersection& its) const {
		EmitterQueryRecord emitterRecord(its.p);
		emitterRecord.ref = ray.o;
		emitterRecord.wi = ray.d;
		emitterRecord.n = its.shFrame.n;
        emitterRecord.uv = its.uv;
		return its.mesh->getEmitter()->eval(emitterRecord); 
	}

    Color3f Li(const Scene* scene, Sampler* sampler, const Ray3f& ray) const {

        Color3f Lo(0.0f);
        Color3f Le_em(0.0f);
        Color3f bsdf_ac(1.0f);
        float russianProb;
        Ray3f bouncyRay(ray);
        bool rayKeepBouncing = true;

        while (rayKeepBouncing)
        {
            // ** RAY INTERSECTION ** //
            Intersection its;

            // Checking if the ray intersects with the scene (OUTPUT: Background)
            if (!scene->rayIntersect(bouncyRay, its)){
                Lo += bsdf_ac * scene->getBackground(bouncyRay);
                break; // Get out of the loop
            } 
            // Checking if the ray intersects with an emitter (OUTPUR: Emitter radiance)
            if (its.mesh->isEmitter()) {
                Lo = bsdf_ac * computeEmitterRadiance(bouncyRay, its);
                break; // Get out of the loop
            }

            // ** BSDF SAMPLING ** //
            // We obtain the new direction wo
            Point2f sample = sampler->next2D();
            BSDFQueryRecord bsdfRecord(its.toLocal(-bouncyRay.d), sample);
            Color3f bsdf_sample = its.mesh->getBSDF()->sample(bsdfRecord, sample);
            bool sampleLightsFlag = (bsdfRecord.measure != EDiscrete);
            float w_mats = sampleLightsFlag ? 0.5f : 1.0f;
            float w_lights = sampleLightsFlag ? 0.5f : 0.0f;

            if (bsdf_sample.isZero() || bsdf_sample.hasNaN()) {
                return Color3f(0.0f); // TODO
            }else{
                bsdf_ac *= bsdf_sample;
            }

            // ** NEXT EVENT CALL DEPENDING ON MATERIAL** //
            if (sampleLightsFlag)
            {
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

                if (!inShadow){
                    float denominator = pdfEmitter * emitterRecord.pdf;
                    if (denominator > Epsilon){
                        BSDFQueryRecord bsdfQR_ls(its.toLocal(-bouncyRay.d), its.toLocal(emitterRecord.wi), its.uv, ESolidAngle);
                        Color3f bsdf = its.mesh->getBSDF()->eval(bsdfQR_ls);
                        Lo += w_lights * bsdf_ac * (Lem * its.shFrame.n.dot(emitterRecord.wi) * bsdf) / denominator;
                    }
                }

            }else{
                /* code */
            }


            // ** RUSSIAN ROULETTE ** //
            // TODO
			russianProb = std::min(bsdf_ac.maxCoeff(), 0.99f);
			if ((sampler->next1D() > russianProb)){
				return Color3f(0.0f); // TODO
			}else{
                bsdf_ac /= russianProb;
            }

            /* UPDATE THE RAY */
            bouncyRay = Ray3f(its.p, its.toWorld(bsdfRecord.wo));            

        }
        
        return Lo;

    }

    std::string toString() const {
        return "Path Tracing NEE []";
    }
};


NORI_REGISTER_CLASS(PathTracingNee, "path_nee");
NORI_NAMESPACE_END