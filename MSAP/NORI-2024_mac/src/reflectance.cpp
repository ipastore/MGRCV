/*
	This file is part of Nori, a simple educational ray tracer
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

#include <nori/reflectance.h>
#include <nori/common.h>
#include <nori/vector.h>
#include <nori/color.h>
#include <nori/frame.h>

NORI_NAMESPACE_BEGIN

Vector3f Reflectance::refract(const Vector3f& wi, const Vector3f& n, float extIOR, float intIOR) {
    
    
    float cosThetaI = Frame::cosTheta(wi);

    float etaI = extIOR, etaT = intIOR;

    /* Swap the indices of refraction if the interaction starts
    at the inside of the object */
    if (cosThetaI < 0.0f) {
        std::swap(etaI, etaT);
        cosThetaI = -cosThetaI;
    }

    float eta = etaI / etaT;

    /* Using Snell's law, calculate the squared sine of the
       angle between the normal and the transmitted ray */
    float sinThetaTSqr = eta * eta * (1 - cosThetaI * cosThetaI);
    float cosThetaT = std::sqrt(1.0f - sinThetaTSqr);

    return Vector3f(-wi[0] * eta, -wi[1] * eta, (wi[2] > 0) ? -cosThetaT : cosThetaT);




}



float Reflectance::fresnel(float cosThetaI, float extIOR, float intIOR) {
    float etaI = extIOR, etaT = intIOR;

    if (extIOR == intIOR)
        return 0.0f;

    /* Swap the indices of refraction if the interaction starts
       at the inside of the object */
    if (cosThetaI < 0.0f) {
        std::swap(etaI, etaT);
        cosThetaI = -cosThetaI;
    }

    /* Using Snell's law, calculate the squared sine of the
       angle between the normal and the transmitted ray */
    float eta = etaI / etaT,
        sinThetaTSqr = eta * eta * (1 - cosThetaI * cosThetaI);

    if (sinThetaTSqr > 1.0f)
        return 1.0f;  /* Total internal reflection! */

    float cosThetaT = std::sqrt(1.0f - sinThetaTSqr);

    float Rs = (etaI * cosThetaI - etaT * cosThetaT)
        / (etaI * cosThetaI + etaT * cosThetaT);
    float Rp = (etaT * cosThetaI - etaI * cosThetaT)
        / (etaT * cosThetaI + etaI * cosThetaT);

    return (Rs * Rs + Rp * Rp) / 2.0f;
}


Color3f Reflectance::fresnel(float cosThetaI, const Color3f &R0)
{
    return R0 + (1. - R0) * pow(1. - cosThetaI, 5.);
}


float Reflectance::G1(const Vector3f& wv, const Vector3f &wh, float alpha)
{
    float b = 1. / (alpha * sqrt(1 - wv[2] * wv[2]) / wv[2]);

    if (wv.dot(wh) / wv[2] <= 0)
        return 0;

    if (b < 1.6)
        return (3.535 * b + 2.181 * b * b) / (1 + 2.276 * b + 2.577 * b * b);
    else
        return 1;
}

float Reflectance::BeckmannNDF(const Vector3f& wh, float alpha)
{
    float tan_thetah = sqrt((float)1 - wh[2] * wh[2]) / wh[2];
    
    return exp(-tan_thetah * tan_thetah / (alpha * alpha)) /
        (M_PI * alpha * alpha * wh[2] * wh[2] * wh[2] * wh[2]); 
}

NORI_NAMESPACE_END