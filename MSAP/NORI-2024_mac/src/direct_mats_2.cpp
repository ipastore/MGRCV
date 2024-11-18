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
        // this integrator casts a ray to the scene and uses brdf sampling to compute the direct illumination
        // the estimate computed by this integrator corresponds to: 
        // Lo(x,ωo) ≈ Le(x,ωo) + (1/N)∑((Le(r(x,ω(k)i),−ω(k)i) fr(x,ωo,ω(k)i) cosθ(k)i) / pΩ(ω(k)i))
        Color3f Lo(0.0f);   // the radiance we will return
        /*
            First ray
        */
        // check if the ray intersects with anything at all
        Intersection its1;
        if (!scene->rayIntersect(ray, its1)) {
            return scene->getBackground(ray);    // if it doesn't intersect, return the background color (end of the path)
        }
        if (its1.mesh->isEmitter()) {   // if it intersects with an emitter, return the radiance of the emitter (end of the path)
            EmitterQueryRecord emitterQR(its1.p);
            emitterQR.ref = ray.o;
			emitterQR.wi = ray.d;
			emitterQR.n = its1.shFrame.n;
            return its1.mesh->getEmitter()->eval(emitterQR);
        }
        /*
            Second ray
        */
        // sample the brdf
        Point2f sample = sampler->next2D();
        BSDFQueryRecord bsdfQR(its1.toLocal(-ray.d), sample);
        Color3f brdfSample = its1.mesh->getBSDF()->sample(bsdfQR, sample);
        // check if the brdf sample is valid (absorbed or invalid samples are not valid)
        if (brdfSample.isZero() || brdfSample.hasNaN()) {   // if it is not valid, return black
            return Color3f(0.0f);
        }
        // now create a new ray with the sampled direction
        Ray3f ray2(its1.p, its1.toWorld(bsdfQR.wo));
        // check if the ray intersects with anything at all
        Intersection its2;
        if (!scene->rayIntersect(ray2, its2)) {
            // if the bounced ray doesnt intersect with nothing, we will add the background color
            // to the radiance we will return
            Color3f backgroundColor = scene->getBackground(ray2);
            Lo = backgroundColor * brdfSample;// * std::abs(Frame::cosTheta(bsdfQR.wo));
            // since there are only two bounced rays, we can return the radiance
            return Lo;
        }
        // if the ray intersects with an emitter, we will add the radiance of the emitter
        // to the radiance we will return
        if (its2.mesh->isEmitter()) {
            EmitterQueryRecord emitterQR(its2.p);
            emitterQR.ref = ray2.o;
			emitterQR.wi = ray2.d;
			emitterQR.n = its2.shFrame.n;
            // calculate the radiance of the emitter to compute the contribution to the returned radiance
            Color3f Le = its2.mesh->getEmitter()->eval(emitterQR);
            // calculate the cosine foreshortening factor (this is the cosine of the angle between the normal and the ray direction)
            // if the ray direction is in the same direction as the normal, the cosine foreshortening factor will be 1
            // and therefore the contribution will be maximum
            // float cosForeshortening = std::abs(its2.shFrame.n.dot(ray2.d)); // this is the same as the commented line below, since vectors are normalized
            // float cosForeshortening = std::abs(Frame::cosTheta(its2.toLocal(-ray2.d)));
            // add the contribution to the returned radiance
            Lo = Le * brdfSample;// * cosForeshortening;
            // NOTE: okay no cosine since we're already taking that into account in the brdf (i think????)
            // since there are only two bounced rays, we can return the radiance
            return Lo;
        }
        // in this case, the ray intersected with a surface that is not an emitter so we will return black
        return Lo;
    }

    std::string toString() const {
        return "Direct Material Sampling []";
    }
};

NORI_REGISTER_CLASS(DirectMaterialSampling, "direct_mats");
NORI_NAMESPACE_END