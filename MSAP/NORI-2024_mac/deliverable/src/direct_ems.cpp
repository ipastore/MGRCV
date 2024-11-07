#include <nori/warp.h>
#include <nori/integrator.h>
#include <nori/scene.h>
#include <nori/emitter.h>
#include <nori/bsdf.h>

NORI_NAMESPACE_BEGIN

class DirectEmitterSampling : public Integrator {
public:
	DirectEmitterSampling(const PropertyList& props) {
		/* No parameters this time */
	}

	Color3f Li(const Scene* scene, Sampler* sampler, const Ray3f& ray) const {
		Color3f Lo(0.);	// default output value
		Intersection its;	// intersection point

		if (!scene->rayIntersect(ray, its))	// if ray doesnt intersect with scene, assume its background
			return scene->getBackground(ray);
		if (its.mesh->isEmitter()) {	// if the intersection point is an emittter, output the radiance of the emitter
			EmitterQueryRecord emitterRecord(its.p);
			emitterRecord.ref = ray.o;
			emitterRecord.wi = ray.d;
			emitterRecord.n = its.shFrame.n;
			return its.mesh->getEmitter()->eval(emitterRecord); 
		}
        // randomly choose an emitter
        float pdfEmitter;	// this is the probability density of choosing an emitter
		const Emitter* em = scene->sampleEmitter(sampler->next1D(), pdfEmitter);
        // get the radiance of said emitter
		EmitterQueryRecord emitterRecord(its.p);
        Color3f Lem = em->sample(emitterRecord, sampler->next2D(), 0.f);	// sample a point on the emitter and get its radiance
        // create a shadow ray to see if the point is in shadow
        Ray3f shadowRay(its.p, emitterRecord.wi);
        shadowRay.maxt = (emitterRecord.p - its.p).norm();
        // check if the ray intersects with an emitter or if the first intersection in shadows
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

	std::string toString() const {
		return "Direct Emitter Sampling []";
	}
};

NORI_REGISTER_CLASS(DirectEmitterSampling, "direct_ems");
NORI_NAMESPACE_END