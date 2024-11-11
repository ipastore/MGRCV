/*
    This file is part of Nori, a simple educational ray tracer

    Copyright (c) 2015 by Wenzel Jakob
	
	v1 - Dec 01 2020
    v2 - Oct 30 2021
	Copyright (c) 2021 by Adrian Jarabo

    Nori is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License Version 3
    as published by the Free Software Foundation.

    Nori is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program. If not, see <http://www.gnu.org/licenses/>.
*/

#include <nori/bsdf.h>
#include <nori/frame.h>
#include <nori/warp.h>
#include <nori/reflectance.h>
#include <nori/texture.h>

NORI_NAMESPACE_BEGIN

#define KS_THRES 0.

class RoughConductor : public BSDF {
public:
    RoughConductor(const PropertyList& propList) {
        /* RMS surface roughness */
        m_alpha = new ConstantSpectrumTexture(propList.getFloat("alpha", 0.1f));

        /* Reflectance at direction of normal incidence.
           To be used when defining the Fresnel term using the Schlick's approximation*/
        m_R0 = new ConstantSpectrumTexture(propList.getColor("R0", Color3f(0.5f)));
    }

    // Vector3f reflect(const Vector3f &wi, const Vector3f &wh) {
    // return wi - 2 * wi.dot(wh) * wh;
    // }   

    /// Evaluate the BRDF for the given pair of directions
    Color3f eval(const BSDFQueryRecord& bRec) const {
        /* This is a smooth BRDF -- return zero if the measure
        is wrong, or when queried for illumination on the backside */
        if (bRec.measure != ESolidAngle
            || Frame::cosTheta(bRec.wi) <= 0
            || Frame::cosTheta(bRec.wo) <= 0)
            return Color3f(0.0f);
        
    // Compute the half-vector
    Vector3f wh = (bRec.wi + bRec.wo).normalized();

    // Retrieve the roughness parameter alpha from the texture
    float alpha = m_alpha->eval(bRec.uv).getLuminance();
    

    // Calculate the Beckmann normal distribution function (NDF) D(wh)
    float D = Reflectance::BeckmannNDF(wh, alpha);

    float cosThetaI = Frame::cosTheta(bRec.wi);
    Color3f R0 = m_R0->eval(bRec.uv);

    // Calculate the Fresnel term F using Schlicks approximation
    Color3f F = Reflectance::fresnel(cosThetaI, R0);

    // Calculate the shadowing-masking term G using Smiths approximation
    float G = Reflectance::G1(bRec.wi, wh, alpha) * Reflectance::G1(bRec.wo, wh, alpha);

    // Calculate the final BRDF value
    float denominator = 4.0f * Frame::cosTheta(bRec.wi) * Frame::cosTheta(bRec.wo);
   

    return (D * F * G) / denominator;
        // throw NoriException("RoughConductor::eval() is not yet implemented!");
    }

    /// Evaluate the sampling density of \ref sample() wrt. solid angles
    float pdf(const BSDFQueryRecord& bRec) const {
        /* This is a smooth BRDF -- return zero if the measure
        is wrong, or when queried for illumination on the backside */
        if (bRec.measure != ESolidAngle
            || Frame::cosTheta(bRec.wi) <= 0
            || Frame::cosTheta(bRec.wo) <= 0)
            return 0.0f;
        
        // Compute the half-vector
        Vector3f wh = (bRec.wi + bRec.wo).normalized();

        // Retrieve the roughness parameter alpha from the texture
        float alpha = m_alpha->eval(bRec.uv).getLuminance();
        
        return Warp::squareToBeckmannPdf(wh, alpha);

        // throw NoriException("RoughConductor::eval() is not yet implemented!");
    }

    /// Sample the BRDF
    Color3f sample(BSDFQueryRecord& bRec, const Point2f& _sample) const {
        // Note: Once you have implemented the part that computes the scattered
        // direction, the last part of this function should simply return the
        // BRDF value divided by the solid angle density and multiplied by the
        // cosine factor from the reflection equation, i.e.
        // return eval(bRec) * Frame::cosTheta(bRec.wo) / pdf(bRec);
        if (Frame::cosTheta(bRec.wi) <= 0)
            return Color3f(0.0f);

        bRec.measure = ESolidAngle;

        // Retrieve the roughness parameter `alpha` by averaging the color channels of `m_alpha`
       float alpha = m_alpha->eval(bRec.uv).getLuminance();

        // Step 1: Sample the half-vector `wh` using the Beckmann distribution
        Vector3f wh = Warp::squareToBeckmann(_sample, alpha);

        // Step 2: Reflect `wi` around `wh` to obtain the outgoing direction `wo`
        //right direction?
        bRec.wo = bRec.wi - 2 * bRec.wi.dot(wh) * wh;

        // Step 3: Ensure the reflected direction is in the correct hemisphere
        if (Frame::cosTheta(bRec.wo) <= 0) {
            return Color3f(0.0f);
        }

        // Step 4: Evaluate the BRDF value at the sampled direction
        Color3f fr = eval(bRec);

        // Step 5: Calculate the PDF value for the sampled direction
        float pdf_val = pdf(bRec);

        // Step 6: Return the BRDF value weighted by the cosine factor and divided by the PDF
        return fr * Frame::cosTheta(bRec.wo) / pdf_val;

        // throw NoriException("RoughConductor::sample() is not yet implemented!");
    }

    bool isDiffuse() const {
        /* While microfacet BRDFs are not perfectly diffuse, they can be
           handled by sampling techniques for diffuse/non-specular materials,
           hence we return true here */
        return true;
    }

    void addChild(NoriObject* obj, const std::string& name = "none") {
        switch (obj->getClassType()) {
        case ETexture:
            if (name == "R0")
            {
                delete m_R0;
                m_R0 = static_cast<Texture*>(obj);
            }
            else if (name == "alpha")
            {
                delete m_alpha;
                m_alpha = static_cast<Texture*>(obj);
            }
            else
                throw NoriException("RoughConductor::addChild(<%s>,%s) is not supported!",
                    classTypeName(obj->getClassType()), name);
            break;
        default:
            throw NoriException("RoughConductor::addChild(<%s>) is not supported!",
                classTypeName(obj->getClassType()));
        }
    }

    std::string toString() const {
        return tfm::format(
            "RoughConductor[\n"
            "  alpha = %f,\n"
            "  R0 = %s,\n"
            "]",
            m_alpha->toString(),
            m_R0->toString()
        );
    }
private:
    Texture* m_alpha;
    Texture* m_R0;
};


class RoughDielectric : public BSDF {
public:
    RoughDielectric(const PropertyList& propList) {
        /* RMS surface roughness */
        m_alpha = new ConstantSpectrumTexture(propList.getFloat("alpha", 0.1f));

        /* Interior IOR (default: BK7 borosilicate optical glass) */
        m_intIOR = propList.getFloat("intIOR", 1.5046f);

        /* Exterior IOR (default: air) */
        m_extIOR = propList.getFloat("extIOR", 1.000277f);

        /* Tint of the glass, modeling its color */
        m_ka = new ConstantSpectrumTexture(propList.getColor("ka", Color3f(1.f)));
    }


    /// Evaluate the BRDF for the given pair of directions
    Color3f eval(const BSDFQueryRecord& bRec) const {
        /* This is a smooth BSDF -- return zero if the measure is wrong */
        if (bRec.measure != ESolidAngle)
            return Color3f(0.0f);


        throw NoriException("RoughDielectric::eval() is not yet implemented!");
    }

    /// Evaluate the sampling density of \ref sample() wrt. solid angles
    float pdf(const BSDFQueryRecord& bRec) const {
        /* This is a smooth BSDF -- return zero if the measure is wrong */
        if (bRec.measure != ESolidAngle)
            return 0.0f;

        throw NoriException("RoughDielectric::eval() is not yet implemented!");
    }

    /// Sample the BRDF
    Color3f sample(BSDFQueryRecord& bRec, const Point2f& _sample) const {
        // Note: Once you have implemented the part that computes the scattered
        // direction, the last part of this function should simply return the
        // BRDF value divided by the solid angle density and multiplied by the
        // cosine factor from the reflection equation, i.e.
        // return eval(bRec) * Frame::cosTheta(bRec.wo) / pdf(bRec);
        bRec.measure = ESolidAngle;

        throw NoriException("RoughDielectric::sample() is not yet implemented!");
    }

    bool isDiffuse() const {
        /* While microfacet BRDFs are not perfectly diffuse, they can be
           handled by sampling techniques for diffuse/non-specular materials,
           hence we return true here */
        return true;
    }

    void addChild(NoriObject* obj, const std::string& name = "none") {
        switch (obj->getClassType()) {
        case ETexture:
            if (name == "m_ka")
            {
                delete m_ka;
                m_ka = static_cast<Texture*>(obj);
            }
            else if (name == "alpha")
            {
                delete m_alpha;
                m_alpha = static_cast<Texture*>(obj);
            }
            else
                throw NoriException("RoughDielectric::addChild(<%s>,%s) is not supported!",
                    classTypeName(obj->getClassType()), name);
            break;
        default:
            throw NoriException("RoughDielectric::addChild(<%s>) is not supported!",
                classTypeName(obj->getClassType()));
        }
    }

    std::string toString() const {
        return tfm::format(
            "RoughDielectric[\n"
            "  alpha = %f,\n"
            "  intIOR = %f,\n"
            "  extIOR = %f,\n"
            "  ka = %s,\n"
            "]",
            m_alpha->toString(),
            m_intIOR,
            m_extIOR,
            m_ka->toString()
        );
    }
private:
    float m_intIOR, m_extIOR;
    Texture* m_alpha;
    Texture* m_ka;
};



class RoughSubstrate : public BSDF {
public:
    RoughSubstrate(const PropertyList &propList) {
        /* RMS surface roughness */
        m_alpha = new ConstantSpectrumTexture(propList.getFloat("alpha", 0.1f));

        /* Interior IOR (default: BK7 borosilicate optical glass) */
        m_intIOR = propList.getFloat("intIOR", 1.5046f);

        /* Exterior IOR (default: air) */
        m_extIOR = propList.getFloat("extIOR", 1.000277f);

        /* Albedo of the diffuse base material (a.k.a "kd") */
        m_kd = new ConstantSpectrumTexture(propList.getColor("kd", Color3f(0.5f)));
    }


    /// Evaluate the BRDF for the given pair of directions
    Color3f eval(const BSDFQueryRecord &bRec) const {
        /* This is a smooth BRDF -- return zero if the measure
        is wrong, or when queried for illumination on the backside */
        if (bRec.measure != ESolidAngle
            || Frame::cosTheta(bRec.wi) <= 0
            || Frame::cosTheta(bRec.wo) <= 0)
            return Color3f(0.0f);
        
        // Half-vector for reflection
    Vector3f wh = (bRec.wi + bRec.wo).normalized();

    // Retrieve the roughness parameter `alpha` from the texture, averaged over color channels
    float alpha = m_alpha->eval(bRec.uv).getLuminance();

    // Fresnel term for diffuse component (Ashikhmin-Shirley model)
    float etaRatio = (m_extIOR - m_intIOR) / (m_extIOR + m_intIOR);
    float F_diff = etaRatio * etaRatio;  // Monochromatic Fresnel term

    // Ashikhmin-Shirley diffuse term
    Color3f kd = m_kd->eval(bRec.uv);  // Albedo (diffuse color)
    float cosThetaI = Frame::cosTheta(bRec.wi);
    float cosThetaO = Frame::cosTheta(bRec.wo);
    
    Color3f f_diff = (28.0f * kd / (23.0f * M_PI)) *
        (1.0f - F_diff) *
        (1.0f - pow(1.0f - 0.5f * cosThetaI, 5.0f)) *
        (1.0f - pow(1.0f - 0.5f * cosThetaO, 5.0f));
    
    // Beckmann NDF for the half-vector `wh`
    float D = Reflectance::BeckmannNDF(wh, alpha);

    // Fresnel term for specular reflection using Schlick's approximation
    float cosThetaH = bRec.wi.dot(wh);  // cos(theta) between `wi` and half-vector `wh`
    Color3f F_spec = Reflectance::fresnel(cosThetaH, kd);

    // Shadowing-masking term (using G1 for both directions)
    float G = Reflectance::G1(bRec.wi, wh, alpha) * Reflectance::G1(bRec.wo, wh, alpha);

    // Specular component of the microfacet BRDF
    float denominator = 4.0f * cosThetaI * cosThetaO;
    float f_mf = D * F_spec[0] * G / denominator;  // Using F_spec[0] for monochromatic

    // Return the combined diffuse and specular components
    return f_diff + f_mf;


		// throw NoriException("RoughSubstrate::eval() is not yet implemented!");
	}

    /// Evaluate the sampling density of \ref sample() wrt. solid angles
    float pdf(const BSDFQueryRecord &bRec) const {
        /* This is a smooth BRDF -- return zero if the measure
       is wrong, or when queried for illumination on the backside */
        if (bRec.measure != ESolidAngle
            || Frame::cosTheta(bRec.wi) <= 0
            || Frame::cosTheta(bRec.wo) <= 0)
            return 0.0f;

        float p_mf = Reflectance::fresnel(Frame::cosTheta(bRec.wi), m_extIOR, m_intIOR);
        float p_diff = 1 - p_mf;
        // now we need to calculate both pdfs and return the weighted sum
        Vector3f wh = (bRec.wi + bRec.wo).normalized();         // wh is the half vector of the in and out directions
        float alpha = m_alpha->eval(bRec.uv).getLuminance();    // alpha param is defining the roughness of the surface
        float pdf_mf = Warp::squareToBeckmannPdf(wh, alpha);
        float pdf_diff = Warp::squareToCosineHemispherePdf(bRec.wo);
        return (p_mf * pdf_mf) + (p_diff * pdf_diff);   // return the weighted sum of the pdfs
		// throw NoriException("RoughSubstrate::eval() is not yet implemented!");
    }

    /// Sample the BRDF
    Color3f sample(BSDFQueryRecord &bRec, const Point2f &_sample) const {
        // Note: Once you have implemented the part that computes the scattered
        // direction, the last part of this function should simply return the
        // BRDF value divided by the solid angle density and multiplied by the
        // cosine factor from the reflection equation, i.e.
        // return eval(bRec) * Frame::cosTheta(bRec.wo) / pdf(bRec);
        if (Frame::cosTheta(bRec.wi) <= 0)
            return Color3f(0.0f);

        bRec.measure = ESolidAngle;

        // choose one component using russian roulette based on the F value
        float alpha = m_alpha->eval(bRec.uv).getLuminance();
        // compute the fresnel term over the surface normal
        float fresnel = Reflectance::fresnel(Frame::cosTheta(bRec.wi), m_extIOR, m_intIOR);
        float random_number = std::rand() / (float)RAND_MAX;

        if (random_number < fresnel) {
            // if microfacet, use beckmann distribution to sample the  microfacet normal
            Vector3f wh = Warp::squareToBeckmann(_sample, alpha);   // this is the microfacet normal
            // calculate the outgoing direction
            bRec.wo = ((2.0f * bRec.wi.dot(wh) * wh) - bRec.wi);    // this is the outgoing direction
        } else {
            // if diffuse, use cosine weighted hemisphere to sample the diffuse normal
            bRec.wo = Warp::squareToCosineHemisphere(_sample);      // this is the outgoing direction
        }
        // return the weight of the sample
        return eval(bRec) * Frame::cosTheta(bRec.wo) / pdf(bRec);

		// throw NoriException("RoughSubstrate::sample() is not yet implemented!");
	}

    bool isDiffuse() const {
        /* While microfacet BRDFs are not perfectly diffuse, they can be
           handled by sampling techniques for diffuse/non-specular materials,
           hence we return true here */
        return true;
    }

    void addChild(NoriObject* obj, const std::string& name = "none") {
        switch (obj->getClassType()) {
        case ETexture:
            if (name == "kd")
            {
                delete m_kd;
                m_kd = static_cast<Texture*>(obj);
            }
            else if (name == "alpha")
            {
                delete m_alpha;
                m_alpha = static_cast<Texture*>(obj);
            }
            else 
                throw NoriException("RoughSubstrate::addChild(<%s>,%s) is not supported!",
                    classTypeName(obj->getClassType()), name);
            break;
        default:
            throw NoriException("RoughSubstrate::addChild(<%s>) is not supported!",
                classTypeName(obj->getClassType()));
        }
    }

    std::string toString() const {
        return tfm::format(
            "RoughSubstrate[\n"
            "  alpha = %f,\n"
            "  intIOR = %f,\n"
            "  extIOR = %f,\n"
            "  kd = %s,\n"
            "]",
            m_alpha->toString(),
            m_intIOR,
            m_extIOR,
            m_kd->toString()
        );
    }
private:
    float m_intIOR, m_extIOR;
    Texture* m_alpha;
    Texture* m_kd;
};

NORI_REGISTER_CLASS(RoughConductor, "roughconductor");
NORI_REGISTER_CLASS(RoughDielectric, "roughdielectric");
NORI_REGISTER_CLASS(RoughSubstrate, "roughsubstrate");

NORI_NAMESPACE_END
