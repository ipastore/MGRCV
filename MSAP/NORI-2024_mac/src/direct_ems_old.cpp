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
            Lo += emitter->eval(emitterRecord);
        }
        
        // Obtenemos todas los emisores de la escena
        // const std::vector<Emitter*> lights = scene->getLights();
        

        // Número de muestras por píxel, ajustable según la calidad deseada
        int N = 64;
        // Para acumular la radiancia total de las muestras
        Color3f accumulatedRadiance(0.0f);

        // Iteramos sobre el numero de sample por cada emisor.
        for (int i = 0; i < N; ++i) {
            
            // Paso 1: Muestrear un emisor aleatorio en la escena
            float pdfEmitter; // PDF de seleccionar este emisor
            const Emitter* em = scene->sampleEmitter(sampler->next1D(), pdfEmitter);
            if (pdfEmitter == 0) continue; // Si la PDF es cero (no hay emisor), sanitizacion por dividir por 0.

            // Paso 2: Muestrear un punto en el emisor seleccionado
            EmitterQueryRecord emitterRecord(its.p);  // Inicializamos el sample point a nuestro punto de referncia para la muestra.
            // Conseguimos la muestra del emisor
            Color3f Le = em->sample(emitterRecord, sampler->next2D(), 0.f);

            // Paso 3: Comprobación de visibilidad mediante un shadowray
            Vector3f lightDir = (emitterRecord.p - its.p).normalized(); // Direccion del emisor a la interseccion
            Ray3f shadowRay(its.p, lightDir);
            if (scene->rayIntersect(shadowRay)) continue; // Si la luz está bloqueada, omitir esta muestra

            // Paso 4: Evaluar el BSDF y el término de coseno
            BSDFQueryRecord bsdfRecord(its.toLocal(-ray.d), its.toLocal(lightDir), its.uv, ESolidAngle);
            Color3f bsdfVal = its.mesh->getBSDF()->eval(bsdfRecord);
            float cosTheta = std::max(0.0f, its.shFrame.n.dot(lightDir));

            if (emitterRecord.pdf == 0) continue;            // Sanity check

            // Paso 5: Acumular la contribución de esta muestra
            accumulatedRadiance += Le * bsdfVal * cosTheta /  (pdfEmitter*emitterRecord.pdf);
        }
            

        // Paso FINAL: Tomamos el promedio de las muestras dividiendo entre `numSamples`
        Lo += accumulatedRadiance / float(N);          
        
        return Lo;
    }

    std::string toString() const {
        return "Direct Emitter Sampling Integrator []";
    }
};

NORI_REGISTER_CLASS(DirectEmitterSampling, "direct_ems");

NORI_NAMESPACE_END
