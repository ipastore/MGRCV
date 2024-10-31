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

Point2f Warp::squareToUniformTriangle(const Point2f& sample) {

    // Reflect the point into the triangle if necessary
    if (sample.x() + sample.y() > 1.0f) {
        return Point2f(1.0f - sample.x(), 1.0f - sample.y());
    }

    return sample;
}

float Warp::squareToUniformTrianglePdf(const Point2f& p) {
    return (p.x() >= 0.0f && p.y() >= 0.0f && p.x() + p.y() <= 1.0f) ? 2.0f : 0.0f;
}

Vector3f Warp::squareToUniformSphere(const Point2f& sample) {

    // Calculate spherical coordinates
    float phi = 2.0f * M_PI * sample.x();           // Azimuthal angle, ranges from 0 to 2pi
    float theta = std::acos(1.0f - 2.0f * sample.y()); // Polar angle, ranges from 0 to pi

    // Convert spherical coordinates to Cartesian coordinates (x, y, z)
    float x = std::sin(theta) * std::cos(phi);
    float y = std::sin(theta) * std::sin(phi);
    float z = std::cos(theta);

    // Return the 3D point on the unit sphere
    return Vector3f(x, y, z);
}

float Warp::squareToUniformSpherePdf(const Vector3f &v) {
    // Check if the point lies on the unit sphere (approximately, due to float precision)
    if (std::abs(v.norm() - 1.0f) < 1e-4f) {
        return 1.0f / (4.0f * M_PI); // PDF for uniform sampling on a sphere
    }
    return 0.0f; // Points outside the unit sphere should have a PDF of zero}
}

Vector3f Warp::squareToUniformHemisphere(const Point2f &sample) {
    // Calculate the azimuthal and polar angles
    float phi = 2.0f * M_PI * sample.x();          // Azimuthal angle
    float theta = std::acos(sample.y());     // Correct polar angle for hemisphere

    // Convert spherical coordinates to Cartesian coordinates (x, y, z)
    float x = std::sin(theta) * std::cos(phi);
    float y = std::sin(theta) * std::sin(phi);
    float z = std::cos(theta);

    // Return the 3D point on the unit hemisphere
    return Vector3f(x, y, z);
}

float Warp::squareToUniformHemispherePdf(const Vector3f &v) {
    // Check if the point lies on the unit sphere (approximately, due to float precision)
    if (std::abs(v.norm() - 1.0f) < 1e-4f && v.z() >= 0.0f) {
        return 1.0f / (2.0f * M_PI); // PDF for uniform sampling on a sphere
    }
    return 0.0f; // Points outside the unit sphere should have a PDF of zero}
}

Vector3f Warp::squareToCosineHemisphere(const Point2f &sample) {
    float r_d = std::sqrt(sample.x());
    float theta_d = 2 * M_PI * sample.y();

    float theta = theta_d;
    float sigma = std::asin(r_d);

    float x = r_d * std::cos(theta_d);
    float y = r_d * std::sin(theta_d);
    float z = std::sqrt(1.0f - r_d*r_d);

    return Vector3f(x,y,z);
}

float Warp::squareToCosineHemispherePdf(const Vector3f &v) {
// Ensure that the point lies on the unit hemisphere (norm ~ 1) and z >= 0
    if ((v.norm() - 1.0f) < 1e-4f && v.z() >= 0.0f) {
        return v.z() / M_PI; // PDF for cosine-weighted sampling
    }
    return 0.0f;
}


Vector3f Warp::squareToBeckmann(const Point2f &sample, float alpha) {
    throw NoriException("Warp::squareToBeckmann() is not yet implemented!");
}

float Warp::squareToBeckmannPdf(const Vector3f &m, float alpha) {
    throw NoriException("Warp::squareToBeckmannPdf() is not yet implemented!");
}

NORI_NAMESPACE_END
