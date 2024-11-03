/*
    This file is part of Nori, a simple educational ray tracer

    Copyright (c) 2015 by Wenzel Jakob

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

#include <nori/warp.h>
#include <nori/vector.h>
#include <nori/frame.h>

NORI_NAMESPACE_BEGIN

Point2f Warp::squareToUniformSquare(const Point2f &sample) {
        return sample;
}

float Warp::squareToUniformSquarePdf(const Point2f &sample) {
    return ((sample.array() >= 0).all() && (sample.array() <= 1).all()) ? 1.0f : 0.0f;
}

Point2f Warp::squareToTent(const Point2f &sample) {
    throw NoriException("Warp::squareToTent() is not yet implemented!");
}

float Warp::squareToTentPdf(const Point2f &p) {
    throw NoriException("Warp::squareToTentPdf() is not yet implemented!");
}

Point2f Warp::squareToUniformDisk(const Point2f &sample) {
    float r = sqrt(sample.x());
    float theta = 2 * M_PI * sample.y();
    return Point2f(r*cos(theta),r*sin(theta));
}

float Warp::squareToUniformDiskPdf(const Point2f &p) {
    if (sqrt(pow(p.x(), 2) + pow(p.y(), 2)) > 1) {
        return 0.0f;
    }
    else {
        return 1/M_PI;
    }
}

Point2f Warp::squareToUniformTriangle(const Point2f& sample) {
    if (sample.x() + sample.y() > 1.0f) {
        return Point2f(1.0f - sample.x(), 1.0f - sample.y());
    }
    else {
        return sample;
    }
    
}

float Warp::squareToUniformTrianglePdf(const Point2f& p) {
    if (p.x() + p.y() > 1.0f) {
        return 0.0f;
    }
    else {
        return 2.0f;    // 1 / 0.5f = 2.0f
    }
}


Vector3f Warp::squareToUniformSphere(const Point2f &sample) {
    float theta = acos(1 - 2 * sample.x()); 
    float phi = 2 * M_PI * sample.y();
    float x = sin(theta) * cos(phi);
    float y = sin(theta) * sin(phi);
    float z = cos(theta);
    // 3. Return the 3D point
    return Vector3f(x, y, z);
    
}

float Warp::squareToUniformSpherePdf(const Vector3f &v) {
    float area = 4 * M_PI;
    return 1 / area;
}

Vector3f Warp::squareToUniformHemisphere(const Point2f &sample) {
    float theta = acos(1 - 2 * sample.x());
    float phi = 2 * M_PI * sample.y();
    float x = sin(theta) * cos(phi);
    float y = sin(theta) * sin(phi);
    float z = cos(theta);
    if (z < 0.0f) {
        z = -z;
    }
    return Vector3f(x, y, z);
}

float Warp::squareToUniformHemispherePdf(const Vector3f &v) {
    if (v.z() < 0.0f) {
        return 0.0f;        
    }
    else {        
        return 1 / (2.0f * M_PI);
    }
    
}

Vector3f Warp::squareToCosineHemisphere(const Point2f &sample) {
    float theta = asin(sqrt(sample.x()));
    float phi = 2 * M_PI * sample.y();
    float x = sin(theta) * cos(phi);
    float y = sin(theta) * sin(phi);
    float z = cos(theta);
    if (z < 0.0f) {
        z = -z;
    }
    return Vector3f(x, y, z);
    
}

float Warp::squareToCosineHemispherePdf(const Vector3f &v) {
    float p = v.z() / ( M_PI); //v.z()=cos(theta)
    if (v.z() < 0.0f) {
        return 0.0f;
    }
    else {
        return p;
    }


}

Vector3f Warp::squareToBeckmann(const Point2f &sample, float alpha) {
    // throw NoriException("Warp::squareToBeckmann() is not yet implemented!");
    // Beckmann half-vector distribution is defined by:
    // D(w_h) = ((exp(-tan^2(theta_h)/alpha^2))/(M_PI*alpha^2*cos^4(theta_h)))
    // with alpha a user-defined roughness parameter and theta_h the half-angle defining the angle between the 
    // half-vector w_h and the north pole of the sphere.
    float phi = M_PI * 2 * sample.x();
    float theta = atan(sqrt(-alpha * alpha * log(1 - sample.y())));
    float cosPhi = cos(phi);
    float sinPhi = sin(phi);
    float cosTheta = cos(theta);
    float sinTheta = sin(theta);
    float x = sinTheta * cosPhi;
    float y = sinTheta * sinPhi;
    float z = cosTheta;
    return Vector3f(x, y, z);
}

float Warp::squareToBeckmannPdf(const Vector3f &m, float alpha) {
    // throw NoriException("Warp::squareToBeckmannPdf() is not yet implemented!");
    if (m.z() <= 0)
        return 0;
    float alpha2 = alpha * alpha;
    float cosTheta = m.z();
    float tanTheta2 = (m.x() * m.x() + m.y() * m.y()) / (cosTheta * cosTheta);
    float cosTheta3 = cosTheta * cosTheta * cosTheta;
    float azimuthal = INV_PI;
    float longitudinal = exp(-tanTheta2 / alpha2) / (alpha2 * cosTheta3);
    return azimuthal * longitudinal;
}

NORI_NAMESPACE_END
