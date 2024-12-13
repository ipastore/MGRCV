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

    Color3f Li(const Scene *scene, Sampler *sampler, const Ray3f &ray) const {
        return tracePath(scene, sampler, ray, Color3f(1.0f), 0, true);
    }

    Color3f sampleLightNEE(const Scene *scene, Sampler *sampler, const Ray3f &bouncyRay, const Intersection &bounceIntersection) const {

        // ** EMISSION SAMPLING ** //
        float pdfEmitter;
        const Emitter *em = scene->sampleEmitter(sampler->next1D(), pdfEmitter);
        Color3f Les;

        // ** CHECKING IF THERE IS ANY EMITTER ** //
        if (em) {
            EmitterQueryRecord emitterRecord(bounceIntersection.p);
            Color3f Lem = em->sample(emitterRecord, sampler->next2D(), 0.f);

            // Generation of a shadow ray to check visibility of the emitter by the intersection point
            Ray3f shadowRay(bounceIntersection.p, emitterRecord.wi);
            shadowRay.maxt = (emitterRecord.p - bounceIntersection.p).norm();

            // ** CHECKING IF THE INTERSECTION POINT IS IN SHADOW ** //
            if (!scene->rayIntersect(shadowRay)) {
                float denominator = pdfEmitter * emitterRecord.pdf;
                if (denominator > Epsilon) {
                    BSDFQueryRecord bsdfQR_ls(bounceIntersection.toLocal(-bouncyRay.d), bounceIntersection.toLocal(emitterRecord.wi), bounceIntersection.uv, ESolidAngle);
                    Color3f bsdf = bounceIntersection.mesh->getBSDF()->eval(bsdfQR_ls);
                    if (bsdf.hasNaN()) {
                        return Color3f(0.0f);
                    }
                    
                    Les = (Lem * bounceIntersection.shFrame.n.dot(emitterRecord.wi) * bsdf) / denominator;
                    // std::cout << "Les: " << Les << std::endl;
                    return Les;
                }

            }else{ // If the intersection point is in shadow, return black
                return Color3f(0.0f);
                std::cout << "The intersection point is in shadow for the emitter sampled" << std::endl;
            }

        }else{ // If there is no emitter, return black
            std::cout << "No emitter found" << std::endl;
            return Color3f(0.0f);
        }
        return Color3f(0.0f);
    }

    Color3f tracePath(const Scene *scene, Sampler *sampler, const Ray3f &bouncyRay, Color3f throughput, int n_bounces, const bool prevMatIsSmooth) const {
        
        Color3f Lo(0.0f);

        // // ************ MAX NUMBER OF BOUNCES ALLOWED ************ //
        // if(n_bounces > 1000) {
        //     return Color3f(0.0f);
        // }

        // ************ BOUNCE RAY INTERSECTION ************ //
        Intersection bounceIntersection;

        // FIRST CASE: Ray doesnt intersect with anything
        if (!scene->rayIntersect(bouncyRay, bounceIntersection)) {
            return throughput * scene->getBackground(bouncyRay);
            // TODO: Necesitamos poner el throughput? -- YES
        }

        // SECOND CASE: Intersection is an emitter
        if (bounceIntersection.mesh->isEmitter()) {
            if (prevMatIsSmooth){
            return throughput * computeEmitterRadiance(bouncyRay, bounceIntersection);
                // TODO: Necesitamos poner el throughput? -- YES
            }else{
                return Color3f(0.0f);
            }            
        }

        // THIRD CASE: Intersection is an NON emitter material

        // ************ BSDF SAMPLING ************ //
        Point2f sample = sampler->next2D();
        BSDFQueryRecord bsdfQR(bounceIntersection.toLocal(-bouncyRay.d), sample);
        Color3f bsdfSample = bounceIntersection.mesh->getBSDF()->sample(bsdfQR, sample);
        bool currentMaterialPerfSmooth = (bsdfQR.measure == EDiscrete);
        
        if (bsdfSample.isZero() || bsdfSample.hasNaN()) {
            return Lo;
        }


        // CASE 1: Material is perfect smooth --> No need to sample lights
        // CASE 2: Material is not perfect smooth --> Need to sample lights. Lo term added
        if (!currentMaterialPerfSmooth){
            Lo += throughput * sampleLightNEE(scene, sampler, bouncyRay, bounceIntersection);
            // TODO: Necesitamos poner el throughput? --> YES
        }

        throughput *= bsdfSample;


        // ** RUSSIAN ROULETTE ** //
        if (n_bounces > 2)
        {
            float russianProb = std::min(throughput.maxCoeff(), 0.99f);
            if (sampler->next1D() > russianProb) {
                return Color3f(0.0f);
            }
            throughput /= russianProb;  // Update throughput using the russian roulette probability
            // TODO: Actualizamos el throughput con el valor de la probabilidad de la ruleta rusa?
        }

        // ******** RECURSIVE CALL ******** //
        // Independent of the material contribution, tracePath is still called
        Ray3f nextRay(bounceIntersection.p, bounceIntersection.toWorld(bsdfQR.wo));
        Lo += tracePath(scene, sampler, nextRay, throughput, n_bounces + 1, currentMaterialPerfSmooth);


        // ******** BACKPROPAGATION WHEN RECURSION HAS FINISHED********* //
        return Lo;

    }

    std::string toString() const {
        return "Path Tracing NEE  []";
    }
};

NORI_REGISTER_CLASS(PathTracingNee, "path_nee");
NORI_NAMESPACE_END
