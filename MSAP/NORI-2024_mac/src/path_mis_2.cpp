#include <nori/warp.h>
#include <nori/integrator.h>
#include <nori/scene.h>
#include <nori/emitter.h>
#include <nori/bsdf.h>

NORI_NAMESPACE_BEGIN

class PathTracingMIS : public Integrator {
public:
	PathTracingMIS(const PropertyList& props) {
		/* No parameters this time */
	}

    Color3f Li(const Scene* scene, Sampler* sampler, const Ray3f& ray) const {
        Color3f Lo(0.0f);   // the radiance we will return
        int depth = 1;
        Color3f throughput(1.0f);
        Ray3f og_ray(ray);
        float survivalProb;
        Intersection its_og;
        if (!scene->rayIntersect(og_ray, its_og)) { // if no intersection, return background color
            return scene->getBackground(og_ray);
        }
        if (its_og.mesh->isEmitter()) {    // if the intersection is an emitter, add the contribution
                EmitterQueryRecord emitterQR(its_og.p);
                emitterQR.n = its_og.shFrame.n;
                emitterQR.ref = og_ray.o;
                emitterQR.uv = its_og.uv;
                emitterQR.wi = og_ray.d;
                emitterQR.dist = its_og.t;
                return its_og.mesh->getEmitter()->eval(emitterQR);
        }
        while (true) {
            // first, get the next ray (and therefore the next intersection) via BSDF sampling
            BSDFQueryRecord bsdfQR_og(its_og.toLocal(-og_ray.d), sampler->next2D());
            Color3f bsdf_og = its_og.mesh->getBSDF()->sample(bsdfQR_og, sampler->next2D());
            if (bsdf_og.isZero() || bsdf_og.hasNaN()) {
                break;
            }
            throughput *= bsdf_og;
            // check if the og intersection is delta
            bool isDelta = bsdfQR_og.measure == EDiscrete;
            // generate the new ray
            Ray3f ray_new(its_og.p, its_og.toWorld(bsdfQR_og.wo));
            Intersection its_new;
            if (!scene->rayIntersect(ray_new, its_new)) {
                Color3f backgroundColor = scene->getBackground(ray_new);
                Lo += backgroundColor * throughput;
                break;
            }
            // p_mat_mat is the probability of sampling the material in this direction
            float p_mat_mat = its_new.mesh->getBSDF()->pdf(bsdfQR_og);
            // p_mat_em is the prob of having sampled the emitter
            float p_mat_em = 0.0f;
            float w_mat = 0.0f;
            if (its_new.mesh->isEmitter()) {
                EmitterQueryRecord emitterQR(its_new.p);
                BSDFQueryRecord bsdfQR_bs(its_new.toLocal(-ray_new.d), sampler->next2D());
                emitterQR.wi = ray_new.d;
                emitterQR.n = its_new.shFrame.n;
                emitterQR.uv = its_new.uv;
                emitterQR.dist = its_new.t;
                // this is the prob of sampling the emitter in this direction
                p_mat_em = its_new.mesh->getEmitter()->pdf(emitterQR);
                if (isDelta)
                    w_mat = 1.0f;
                else {
                    float w_mat_den = p_mat_mat + p_mat_em;
                    if (w_mat_den > Epsilon) {
                        w_mat = p_mat_mat / w_mat_den;
                    }
                }
                Lo += w_mat * throughput * its_new.mesh->getEmitter()->eval(emitterQR);
                break;
            }
            if (!isDelta){
                float pdf_emitter;
                const Emitter* em = scene->sampleEmitter(sampler->next1D(), pdf_emitter);
                EmitterQueryRecord emitterQR_ls(its_og.p);
                Color3f Le = em->sample(emitterQR_ls, sampler->next2D(), 0.0f);
                Ray3f ray_shadow(its_og.p, emitterQR_ls.wi);
                ray_shadow.maxt = (emitterQR_ls.p - its_og.p).norm();
                Intersection its_shadow;
                bool in_shadow = scene->rayIntersect(ray_shadow, its_shadow);
                if (!in_shadow || (its_shadow.t >= (emitterQR_ls.dist - Epsilon))) {
                    // this BSDFQueryRecord will be the one for the light sampling (contains shadow ray direction)
                    BSDFQueryRecord bsdfQR_ls(its_og.toLocal(-og_ray.d), its_og.toLocal(emitterQR_ls.wi), its_og.uv, ESolidAngle);
                    float ls_den = pdf_emitter * emitterQR_ls.pdf;
                    if (ls_den > Epsilon) {
                        Color3f bsdf = its_og.mesh->getBSDF()->eval(bsdfQR_ls);
                        float pdf_bsdf = its_og.mesh->getBSDF()->pdf(bsdfQR_ls);    // prob of sampling the light direction by BSDF sampling
                        pdf_emitter = em->pdf(emitterQR_ls);    // prob of sampling the light direction by light sampling
                        float w_em_den = pdf_emitter + pdf_bsdf;
                        float w_em = 0.0f;
                        if (w_em_den > Epsilon) {
                            w_em = pdf_emitter / w_em_den;
                        }
                        Color3f L_ls = (Le * its_og.shFrame.n.dot(emitterQR_ls.wi) * bsdf) / ls_den;
                        Lo += w_em * throughput * L_ls;
                    }
                }
            }
            if (depth > 2) {
                survivalProb = std::min(throughput.maxCoeff(), 0.95f);
                if (sampler->next1D() > survivalProb) {
                    break;
                }
                throughput /= survivalProb;
            }
            og_ray = Ray3f(ray_new);
            its_og = Intersection(its_new);
            depth++;
        }
        return Lo;
    }

    std::string toString() const {
        return "Direct Multiple Importance Sampling []";
    }
};

NORI_REGISTER_CLASS(PathTracingMIS, "path_mis");
NORI_NAMESPACE_END