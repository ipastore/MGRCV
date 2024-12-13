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

    Color3f Li(const Scene* scene, Sampler* sampler, const Ray3f& ray) const {
        Color3f Lo(0.0f);   // the radiance we will return
        int depth = 1;
        Color3f throughput(1.0f);
        Ray3f bouncyRay(ray);
        float survivalProb;
        while (true) {
            Intersection its;
            if (!scene->rayIntersect(bouncyRay, its)) {
                // if the ray doesnt intersect with nothing, we will add the background color
                // to the radiance we will return
                Color3f backgroundColor = scene->getBackground(bouncyRay);
                Lo += backgroundColor * throughput;
                break;
            }
            /*
            *   NOW WE HAVE AN INTERSECTION
            */
            Point2f sample = sampler->next2D();
            BSDFQueryRecord bsdfQR(its.toLocal(-bouncyRay.d), sample);
            int sampleLights = (bsdfQR.measure != EDiscrete);
            float w_mats = sampleLights ? 0.5f : 1.0f;
            float w_lights = sampleLights ? 0.5f : 0.0f;
            // if the ray intersects with an emitter, we will add the radiance of the emitter (if it's not perfect smooth)
            if (its.mesh->isEmitter()) {
                sampleLights = false; // THE MATERIAL DOESN'T NECESSARILY NEED TO BE DELTA, BUT WE WILL CONSIDER IT AS ONE
                w_mats = 1.0f;  // THIS IS IN ORDER TO ONLY SAMPLE THE EMMITER ONCE
                EmitterQueryRecord emitterQR(its.p);
                emitterQR.ref = bouncyRay.o;
                emitterQR.wi = bouncyRay.d;
                emitterQR.n = its.shFrame.n;
                emitterQR.uv = its.uv;
                Lo += w_mats * its.mesh->getEmitter()->eval(emitterQR) * throughput;
                break;
            }

            /* BSDF SAMPLING */
            Color3f L_bs(0.0f);
            Color3f bsdfSample = its.mesh->getBSDF()->sample(bsdfQR, sample);

            if (bsdfSample.isZero() || bsdfSample.hasNaN()) {
                break;
            }
            // in any case, we need to update the throughput
            throughput *= bsdfSample;
            
            /* LIGHT SAMPLING */
            // we will only do light sampling if the BSDF is not perfectly smooth
            if (sampleLights) {
                // randomly choose an emitter and add its contribution to the throughput
                float pdflight;	// this is the probability density of choosing a light source
                EmitterQueryRecord emitterQR_ls(its.p);	// add intersection point to emitterRecord
                const Emitter* em = scene->sampleEmitter(sampler->next1D(), pdflight); 		// sample a random light source
                Color3f Le = em->sample(emitterQR_ls, sampler->next2D(), 0.);	// radiance of the light source
                Ray3f shadowRay(its.p, emitterQR_ls.wi); // shadow ray that goes from the intersection point to the light source
                shadowRay.maxt = (emitterQR_ls.p - its.p).norm();	// maxt is the distance between the intersection point and the light source (?)
                // if the shadow ray doesnt intersect with the scene, or if it intersects after the light source, then the point is not in shadow
                Color3f L_ls(0.0f);
                Intersection shadowIts;
                bool inShadow = scene->rayIntersect(shadowRay, shadowIts);
                if (!inShadow) {
                    BSDFQueryRecord bsdfQR_ls(its.toLocal(-bouncyRay.d), its.toLocal(emitterQR_ls.wi), its.uv, ESolidAngle);
                    float denominator = pdflight * emitterQR_ls.pdf;
                    if (denominator > Epsilon){	// to avoid division by 0 (resulting in NaNs and anoying warnings)
                        // emitterQR_ls.dist = its.t;
                        Color3f bsdf = its.mesh->getBSDF()->eval(bsdfQR_ls);
                        // update the color
                        Lo += w_lights * throughput * (Le * its.shFrame.n.dot(emitterQR_ls.wi) * bsdf) / denominator;
                    }
                }
            }
            /* RUSSIAN ROULETTE */
            if (depth > 2) {    // we want to ensure that the path has at least  bounces
                // start the russian roulette
                // max component of the throughput will be the probability of survival (we cap it at 0.95)
                survivalProb = std::min(throughput.maxCoeff(), 0.99f);
                if (sampler->next1D() > survivalProb) { // this is the russian roulette
                    break;  // if the ray dies, we stop the loop
                } else {
                    throughput /= survivalProb; // if the ray survives, we need to update the throughput
                }
            }

            /* UPDATE THE RAY */
            bouncyRay = Ray3f(its.p, its.toWorld(bsdfQR.wo));
            depth++;
        }
        return Lo;
    }

    std::string toString() const {
        return "Path Tracing []";
    }
};

NORI_REGISTER_CLASS(PathTracingNee, "path_nee");
NORI_NAMESPACE_END