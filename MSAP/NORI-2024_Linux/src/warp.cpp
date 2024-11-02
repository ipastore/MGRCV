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
    float r = std::sqrt(sample.x());
    float theta = 2 * M_PI * sample.y();

    float x = r * std::cos(theta);
    float y = r * std::sin(theta);

    return Point2f(x,y);
}

float Warp::squareToUniformDiskPdf(const Point2f &p) {
    if (p.norm() <= 1.0f) {
        return 1.0f / M_PI;
    }else{
        return 0.0f;
    }
}

Point2f squareToUniformTriangle(const Point2f& sample) {
    float sqrtU = std::sqrt(sample.x());
    float u = 1.0f - sqrtU;
    float v = sample.y() * sqrtU;
    return Point2f(u, v);
}

float Warp::squareToUniformTrianglePdf(const Point2f& p) {
    // Check if the point lies within the bounds of the unit triangle
    return (p.x() + p.y() <= 1.0f) ? 2.0f : 0.0f;
}


// Vector3f Warp::squareToUniformSphere(const Point2f& sample) {

//     // Calculate spherical coordinates
//     float phi = 2.0f * M_PI * sample.x();           // Azimuthal angle, ranges from 0 to 2pi
//     float theta = std::acos(1.0f - 2.0f * sample.y()); // Polar angle, ranges from 0 to pi

//     // Convert spherical coordinates to Cartesian coordinates (x, y, z)
//     float x = std::sin(theta) * std::cos(phi);
//     float y = std::sin(theta) * std::sin(phi);
//     float z = std::cos(theta);

//     // Return the 3D point on the unit sphere
//     return Vector3f(x, y, z);
// }

// float Warp::squareToUniformSpherePdf(const Vector3f &v) {
//     // Check if the point lies on the unit sphere (approximately, due to float precision)
//     if (std::abs(v.norm() - 1.0f) < 1e-4f) {
//         return 1.0f / (4.0f * M_PI); // PDF for uniform sampling on a sphere
//     }
//     return 0.0f; // Points outside the unit sphere should have a PDF of zero}
// }

// Vector3f Warp::squareToUniformHemisphere(const Point2f &sample) {
//     // Calculate the azimuthal and polar angles
//     float phi = 2.0f * M_PI * sample.x();          // Azimuthal angle
//     float theta = std::acos(sample.y());     // Correct polar angle for hemisphere

//     // Convert spherical coordinates to Cartesian coordinates (x, y, z)
//     float x = std::sin(theta) * std::cos(phi);
//     float y = std::sin(theta) * std::sin(phi);
//     float z = std::cos(theta);

//     // Return the 3D point on the unit hemisphere
//     return Vector3f(x, y, z);
// }

// float Warp::squareToUniformHemispherePdf(const Vector3f &v) {
//     // Check if the point lies on the unit sphere (approximately, due to float precision)
//     if (std::abs(v.norm() - 1.0f) < 1e-4f && v.z() >= 0.0f) {
//         return 1.0f / (2.0f * M_PI); // PDF for uniform sampling on a sphere
//     }
//     return 0.0f; // Points outside the unit sphere should have a PDF of zero}
// }

// Vector3f Warp::squareToCosineHemisphere(const Point2f &sample) {
//     float r_d = std::sqrt(sample.x());
//     float theta_d = 2 * M_PI * sample.y();

//     float sigma = std::asin(r_d);

//     float x = r_d * std::cos(theta_d);
//     float y = r_d * std::sin(theta_d);
//     float z = std::sqrt(1.0f - r_d*r_d);

//     return Vector3f(x,y,z);
// }

// float Warp::squareToCosineHemispherePdf(const Vector3f &v) {
// // Ensure that the point lies on the unit hemisphere (norm ~ 1) and z >= 0
//     if ((v.norm() - 1.0f) < 1e-4f && v.z() >= 0.0f) {
//         return v.z() / M_PI; // PDF for cosine-weighted sampling
//     }
//     return 0.0f;
// }


// Vector3f Warp::squareToBeckmann(const Point2f &sample, float alpha) {

//     // Step 1: Sample polar angle theta
//     float tanTheta2 = -alpha * alpha * std::log(1.0f - sample.x()); // x in [0, 1]
//     float theta = std::atan(std::sqrt(tanTheta2)); // theta is in [0, pi/2]

//     // Step 2: Sample azimuthal angle phi uniformly
//     float phi = 2.0f * M_PI * sample.y(); // y in [0, 1]

//     // Step 3: Convert spherical coordinates to Cartesian
//     float sinTheta = std::sin(theta);
//     float x = sinTheta * std::cos(phi);
//     float y = sinTheta * std::sin(phi);
//     float z = std::cos(theta);

//     return Vector3f(x, y, z); // Beckmann-sampled direction
//     }

// float Warp::squareToBeckmannPdf(const Vector3f &m, float alpha) {
    
//     // Ensure the vector is on the upper hemisphere
//     if (m.z() <= 0.0f) return 0.0f;

//     // Compute cos(theta) and tan^2(theta)
//     float cosTheta = m.z();
//     float tanTheta2 = (1.0f - cosTheta * cosTheta) / (cosTheta * cosTheta);

//     // Compute the Beckmann PDF
//     float exponent = -tanTheta2 / (alpha * alpha);
//     float pdf = (std::exp(exponent) / (M_PI * alpha * alpha * cosTheta * cosTheta * cosTheta));

//     return pdf;

// }

NORI_NAMESPACE_END
