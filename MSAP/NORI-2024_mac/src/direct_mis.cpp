#include <nori/warp.h>
#include <nori/integrator.h>
#include <nori/scene.h>
#include <nori/emitter.h>
#include <nori/bsdf.h>

NORI_NAMESPACE_BEGIN

class DirectMIS : public Integrator {
public:
	DirectMIS(const PropertyList& props) {
		/* No parameters this time */
	}

    Color3f Li_ma(const Scene* scene, Sampler* sampler, const Ray3f& ray, const Intersection& its) const {

        Color3f Le_ma(0.0f);  // BRDF sampling contribution
        float w_mats = 0.f;
        float p_mat_mat = 0.f, p_em_mat = 0.f;

        // ** SAMPLING THE SECOND RAY ** //

		BSDFQueryRecord bsdfQR(its.toLocal(-ray.d), sampler->next2D());
		Color3f bsdf = its.mesh->getBSDF()->sample(bsdfQR, sampler->next2D());

		// We must check if the BSDF value is valid, if not the output will be zero
		if (bsdf.isValid() || !bsdf.isZero()){
			
            // ** CHECKING SECOND INTERSECTION ** //

            Intersection its2;
            Ray3f second_ray(its.p, its.toWorld(bsdfQR.wo));

            // Checking if the SECOND ray intersects with the scene
            if (!scene->rayIntersect(second_ray, its2)){
                Color3f backgroundColor = scene->getBackground(second_ray);
                Le_ma = backgroundColor * bsdf * std::abs(Frame::cosTheta(bsdfQR.wo));
            }
            
            // Checking if the SECOND ray intersects with an emitter (OUTPUR: Emitter radiance)
            if (its2.mesh->isEmitter()) {	
                EmitterQueryRecord emitterRecord_2(its2.p);
                emitterRecord_2.ref = second_ray.o;
                emitterRecord_2.wi = second_ray.d;
                emitterRecord_2.n = its2.shFrame.n;

                p_em_mat = its2.mesh->getEmitter()->pdf(emitterRecord_2);
                Color3f Le = its2.mesh->getEmitter()->eval(emitterRecord_2); 

                Le_ma =  bsdf * Le;
            }

            p_mat_mat = its.mesh->getBSDF()->pdf(bsdfQR);

            if (p_em_mat + p_mat_mat > Epsilon){
                w_mats = p_mat_mat / (p_em_mat + p_mat_mat);
            }
        }

        return Le_ma * w_mats;        

    }

    Color3f Li_em(const Scene* scene, Sampler* sampler, const Ray3f& ray, const Intersection& its) const {

        Color3f Le_em(0.0f);
        float w_em = 0.f;
        float p_mat_em = 0.f, p_em_em = 0.f;

        // ** RANDOMLY CHOOSING AN EMITER ** //

        float pdfEmitter;
		const Emitter* em = scene->sampleEmitter(sampler->next1D(), pdfEmitter);

        // ** SAMPLING AND EVALUATING A POINT IN THE EMITER ** //

		EmitterQueryRecord emitterRecord(its.p);
        Color3f Lem = em->sample(emitterRecord, sampler->next2D(), 0.f);

        // ** CHECKING IF THE POINT IS IN SHADOW ** //

        Intersection shadow_its;
        Ray3f shadowRay(its.p, emitterRecord.wi);
        shadowRay.maxt = (emitterRecord.p - its.p).norm();

        bool inShadow = scene->rayIntersect(shadowRay, shadow_its);

        if (!inShadow || shadow_its.t >= (emitterRecord.dist - Epsilon)){
            BSDFQueryRecord bsdfQR_ls(its.toLocal(-ray.d), its.toLocal(emitterRecord.wi), its.uv, ESolidAngle);
			Color3f bsdf = its.mesh->getBSDF()->eval(bsdfQR_ls);
            p_em_em = pdfEmitter * emitterRecord.pdf;
            p_mat_em = its.mesh->getBSDF()->pdf(bsdfQR_ls);


            if (p_em_em > Epsilon){	
                emitterRecord.dist = its.t;
                Le_em = (Lem * its.shFrame.n.dot(emitterRecord.wi) * bsdf) / p_em_em;
			}
		}

        if (p_em_em + p_mat_em > Epsilon){
            w_em = p_em_em / (p_em_em + p_mat_em);
        }
        
        return Le_em * w_em;        

    }

	Color3f Li(const Scene* scene, Sampler* sampler, const Ray3f& ray) const {

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

		// ** COMPUTING MATERIAL AND EMITTER SAMPLIGS CONTRIBUTIONS ** //

        Color3f Le_ma = Li_ma(scene, sampler, ray, its);
        Color3f Le_em = Li_em(scene, sampler, ray, its);

        return Le_ma + Le_em;
	}

	std::string toString() const {
		return "Multiple Importance Sampling []";
	}
};

NORI_REGISTER_CLASS(DirectMIS, "direct_mis");
NORI_NAMESPACE_END