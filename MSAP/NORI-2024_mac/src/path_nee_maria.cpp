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

    Color3f Li(const Scene* scene, Sampler* sampler, const Ray3f& ray) const {
        //Color black as default
		Color3f Lo(0.0f); 

		//Find the intersection point that is visible in the request direction
		Intersection its;

		// If ray does not intersect with scene, assume the background
		if (!scene->rayIntersect(ray, its))
			return scene->getBackground(ray);
        
        if (its.mesh->isEmitter()) {   // if it intersects with an emitter, return the radiance of the emitter (end of the path)
            EmitterQueryRecord emitterQR(its.mesh->getEmitter(), its.p, its.p, its.shFrame.n, its.uv);
            return its.mesh->getEmitter()->eval(emitterQR);
        }
        // if it is not an emitter

        // --- Direct Illumination via Emitter Sampling ---

        /// array containing all lights
		const std::vector<Emitter *>& lights = scene->getLights();
		
		size_t n_emitters = lights.size();
		if (n_emitters == 0) {
			return Color3f(0.0f); // No emitters
		}

		//Probablity of choosing a lightsource
		float pdflight = 1.0f / n_emitters;
		
		//The intersection surface it is also an emitter
		EmitterQueryRecord emitterRecord(its.p); 

		//Choose a random emitter light
		const Emitter* em = scene->sampleEmitter(sampler->next1D(), pdflight);
        emitterRecord.emitter = em;

		//Sample a point on the emitter and get the randiance in that direction
		Color3f Le = (em->sample(emitterRecord, sampler->next2D(), 0.));   

		//emitterRecord.wi is the direction vector from the intersection point (its.p) to the light source.
		Vector3f wi = emitterRecord.wi;

		//We create a shadow ray (shadowRay) that starts at the intersection point its.p and goes in the direction of wi
		Ray3f shadowRay(its.p, wi);

		//shadowRay.maxt is set to the distance between the intersection point its.p and the position of the light source emitterRecord.p, normalized by .norm().
		shadowRay.maxt = (emitterRecord.p - its.p).norm() - Epsilon;
        shadowRay.mint = Epsilon;

		Intersection its_sh;
		bool inter_shadow = scene->rayIntersect(shadowRay, its_sh); //Check if the ray intersect with the scene

		if (!inter_shadow) { // No occlusion

			BSDFQueryRecord bsdfRecord(its.toLocal(-ray.d), its.toLocal(emitterRecord.wi), its.uv, ESolidAngle);

			// In direct_ems we sample one lightsource, and we divide Lo by the emitterRecord.pdf and pdflight. 
			Lo += (Le * its.shFrame.n.dot(emitterRecord.wi) * its.mesh->getBSDF()->eval(bsdfRecord)) / (pdflight * emitterRecord.pdf);
            Lo = Lo.clamp();
        } 

        // --- Indirect Illumination via BSDF Sampling ---

        // sample the brdf
        BSDFQueryRecord bsdfQR(its.toLocal(-ray.d), sampler->next2D());
        // BSDF intersection
        Color3f brdfSample = its.mesh->getBSDF()->sample(bsdfQR, sampler->next2D()); 
        // check if the brdf sample is valid 
        if (brdfSample.isZero() || brdfSample.hasNaN()) {   // if it is not valid, return black
            return Lo;
        } 
        // now create a new ray with the sampled direction
        Ray3f rayBSDF(its.p, its.toWorld(bsdfQR.wo));
        // check if the ray intersects with anything at all
        Intersection its_bsdf;

        if (!scene->rayIntersect(rayBSDF, its_bsdf)) {
            return scene->getBackground(rayBSDF) * brdfSample; }
        else if(its_bsdf.mesh->isEmitter() && bsdfQR.measure == EDiscrete) {     // if the ray intersects with an emitter, we will add the radiance of the emitter to the radiance we will return
            EmitterQueryRecord emitterBSDF(its_bsdf.mesh->getEmitter(), its_bsdf.p, its_bsdf.p, its_bsdf.shFrame.n, its_bsdf.uv);
            //emitterBSDF.ref = rayBSDF.o;
            //emitterBSDF.wi = rayBSDF.d;
            //emitterBSDF.n = its_bsdf.shFrame.n;
            
            // calculate the radiance of the emitter to compute the contribution to the returned radiance
            Color3f Le = its_bsdf.mesh->getEmitter()->eval(emitterBSDF);
            Lo = Le * brdfSample;
            return Lo;          
        }

        float r = sampler->next1D();
        float p = brdfSample.maxCoeff();
        if(r > p) 
            return Lo;
        else
            return Li(scene,sampler,rayBSDF) * brdfSample / p ;
   
    }

    std::string toString() const {
        return "Path Tracing []";
    };
};

NORI_REGISTER_CLASS(PathTracingNEE, "path_nee");

NORI_NAMESPACE_END