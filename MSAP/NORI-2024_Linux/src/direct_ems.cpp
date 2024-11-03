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
        
        // Inicializamos Lo(x, wo) a 0. 
        Color3f Lo(0.);

        // Para cada pixel, comprobamos si el rayo intersecta con algo.
        Intersection its;
        if (!scene->rayIntersect(ray, its))
            // Si no intersecta con nada, devolvemos el background y salimos de la funcion
            return scene->getBackground(ray);
        
        // Comprobamos si la interseccion corresponde con un emisor
        if (its.mesh->isEmitter()) {
            const Emitter* emitter = its.mesh->getEmitter(); 
            EmitterQueryRecord emitterRecord(its.p); // Intersection point on the emitter
			emitterRecord.ref = ray.o;               // Reference point (camera position)
			emitterRecord.wi = ray.d;                // Direction vector from 'ref' to 'p'
			emitterRecord.n = its.shFrame.n;         // Normal at the intersection point
            // Para cada interseccion que es un emisor, se le suma el termino 'directly visible radiance'
            // que equivale al termino Le(x, wo) previo a la integracion de Monte Carlo
            return emitter->eval(emitterRecord);
        }

        // Para acumular la radiancia total de las muestras
        // Color3f accumulatedRadiance(0.0f);
            
        // Paso 1: Muestrear un emisor aleatorio en la escena
        float pdfEmitter; // PDF de seleccionar este emisor
        const Emitter* em = scene->sampleEmitter(sampler->next1D(), pdfEmitter);
        // if (pdfEmitter == 0) return 0.f; // Si la PDF es cero (no hay emisor), sanitizacion por dividir por 0.

        // Paso 2: Muestrear un punto en el emisor seleccionado
        EmitterQueryRecord emitterRecord(its.p);  // Inicializamos el sample point a nuestro punto de referncia para la muestra.
        // Conseguimos la muestra del emisor
        Color3f Le = em->sample(emitterRecord, sampler->next2D(), 0.f);

        // Paso 3: Comprobación de visibilidad mediante un shadowray
        // Vector3f lightDir = (emitterRecord.p - its.p).norm(); // Direccion del emisor a la interseccion
        // Ray3f shadowRay(its.p, lightDir);
        // if (scene->rayIntersect(shadowRay)) return 0.f; // Si la luz está bloqueada, omitir esta muestra

        // Paso 4: Evaluar el BSDF y el término de coseno
        // BSDFQueryRecord bsdfRecord(its.toLocal(-ray.d), its.toLocal(lightDir), its.uv, ESolidAngle);
        // Color3f bsdfVal = its.mesh->getBSDF()->eval(bsdfRecord);
        // float cosTheta = std::max(0.0f, its.shFrame.n.dot(lightDir));

        // if (emitterRecord.pdf == 0) return 0.f;            // Sanity check

        // // Paso 5: Acumular la contribución de esta muestra
        // Lo += Le * bsdfVal * cosTheta /  (pdfEmitter*emitterRecord.pdf);
        		// create a shadow ray to see if the point is in shadow
        // Ray3f shadowRay(its.p, emitterRecord.wi);
        Vector3f lightDir = (emitterRecord.p - its.p).normalized(); // Direccion del emisor a la interseccion
        Ray3f shadowRay(its.p, lightDir);
        // shadowRay.maxt = (emitterRecord.p - its.p).norm();
        // check if the ray intersects with an emitter or if the first intersection in shadows
        Intersection shadow_its;
        bool inShadow = scene->rayIntersect(shadowRay, shadow_its);


        if (!inShadow || shadow_its.t >= (emitterRecord.dist - Epsilon)){
            BSDFQueryRecord bsdfQR_ls(its.toLocal(-ray.d), its.toLocal(emitterRecord.wi), its.uv, ESolidAngle);
			
			
			Color3f bsdf = its.mesh->getBSDF()->eval(bsdfQR_ls);
            float denominator = pdfEmitter * emitterRecord.pdf;
            if (denominator > Epsilon){	// to avoid division by 0 (resulting in NaNs and anoying warnings)
                emitterRecord.dist = its.t;
                Lo = (Le * its.shFrame.n.dot(emitterRecord.wi) * bsdf) / denominator;
			}
		}
            

        // Paso FINAL: Tomamos el promedio de las muestras dividiendo entre `numSamples`
        // Lo += accumulatedRadiance / float(N);          
        
        return Lo;
    }

    std::string toString() const {
        return "Direct Emitter Sampling Integrator []";
    }
};

NORI_REGISTER_CLASS(DirectEmitterSampling, "direct_ems");

NORI_NAMESPACE_END
