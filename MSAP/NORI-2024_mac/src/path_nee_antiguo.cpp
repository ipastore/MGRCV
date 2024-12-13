#include <nori/warp.h>
#include <nori/integrator.h>
#include <nori/scene.h>
#include <nori/emitter.h>
#include <nori/bsdf.h>

NORI_NAMESPACE_BEGIN

class PathTracingNee : public Integrator {
public:
    PathTracingNee(const PropertyList &props) {}

    Color3f computeEmitterRadiance(const Ray3f &ray, const Intersection &its) const {
        EmitterQueryRecord emitterRecord(its.p);
        emitterRecord.ref = ray.o;
        emitterRecord.wi = ray.d;
        emitterRecord.n = its.shFrame.n;
        emitterRecord.uv = its.uv;
        return its.mesh->getEmitter()->eval(emitterRecord);
    }

    Color3f tracePath(const Scene *scene, Sampler *sampler, const Ray3f &bouncyRay, Color3f bsdf_ac, int n_bounces) const {
        if (n_bounces > 100){ // Límite de profundidad
            return Color3f(0.0f);
        }

        // ** RAY INTERSECTION ** //
        Intersection its;
        if (!scene->rayIntersect(bouncyRay, its)) {
            return bsdf_ac * scene->getBackground(bouncyRay);
        }

        // ** EMITTER CHECK ** //
        if (its.mesh->isEmitter()) {
            return bsdf_ac * computeEmitterRadiance(bouncyRay, its);
        }

        // ** BSDF SAMPLING ** //
        Point2f sample = sampler->next2D();
        BSDFQueryRecord bsdfRecord(its.toLocal(-bouncyRay.d), sample);
        Color3f bsdf_sample = its.mesh->getBSDF()->sample(bsdfRecord, sample);
        bool sampleLightsFlag = (bsdfRecord.measure != EDiscrete);

        if (bsdf_sample.isZero() || bsdf_sample.hasNaN()) {
            return Color3f(0.0f);
        }

        bsdf_ac *= bsdf_sample;

        // ** NEXT EVENT ESTIMATION ** //

        // ** NEXT RAY ** //
        Color3f Lo(0.0f);
        Intersection nex_its;
        Ray3f nextRay(its.p, its.toWorld(bsdfRecord.wo));
        // bool NEEkeepBouncing = scene->rayIntersect(nextRay, nex_its);
        // if (NEEkeepBouncing) { // To avoid segmentation fault
        //     if (nex_its.mesh->isEmitter()) {
        //         Lo += bsdf_ac * computeEmitterRadiance(nextRay, nex_its);
        //         return Lo;
        //     }
        // }

        if (sampleLightsFlag) {
            float pdfEmitter;
            const Emitter *em = scene->sampleEmitter(sampler->next1D(), pdfEmitter);

            if (em) {
                EmitterQueryRecord emitterRecord(its.p);
                Color3f Lem = em->sample(emitterRecord, sampler->next2D(), 0.f);

                Ray3f shadowRay(its.p, emitterRecord.wi);
                shadowRay.maxt = (emitterRecord.p - its.p).norm();

                Intersection shadow_its;
                if (!scene->rayIntersect(shadowRay, shadow_its)) {
                    float denominator = pdfEmitter * emitterRecord.pdf;
                    if (denominator > Epsilon) {
                        BSDFQueryRecord bsdfQR_ls(its.toLocal(-bouncyRay.d), its.toLocal(emitterRecord.wi), its.uv, ESolidAngle);
                        Color3f bsdf = its.mesh->getBSDF()->eval(bsdfQR_ls);
                        Lo += bsdf_ac * (Lem * its.shFrame.n.dot(emitterRecord.wi) * bsdf) / denominator;
                    }
                }
            }
        }

        // ** RUSSIAN ROULETTE ** //
        if (n_bounces > 2)
        {
            float russianProb = std::min(bsdf_ac.maxCoeff(), 0.99f);
            if (sampler->next1D() > russianProb) {
                return Color3f(0.0f); // TODO
                //TODO
            }
            bsdf_ac /= russianProb;
        }

        // ** RECURSIVE CALL ** //
        Lo += tracePath(scene, sampler, nextRay, bsdf_ac, n_bounces + 1);
        return Lo;
    }

    // Punto de entrada principal
    Color3f Li(const Scene *scene, Sampler *sampler, const Ray3f &ray) const {
        return tracePath(scene, sampler, ray, Color3f(1.0f), 0);
    }

    std::string toString() const {
        return "Path Tracing NEE []";
    }
};

NORI_REGISTER_CLASS(PathTracingNee, "path_nee");
NORI_NAMESPACE_END
