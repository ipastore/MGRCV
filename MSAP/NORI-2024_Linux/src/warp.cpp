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
    throw NoriException("Warp::squareToUniformDisk() is not yet implemented!");
}

float Warp::squareToUniformDiskPdf(const Point2f &p) {
    throw NoriException("Warp::squareToUniformDiskPdf() is not yet implemented!");
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

    // Calculate the azimuthal and polar angles
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
    throw NoriException("Warp::squareToUniformSpherePdf() is not yet implemented!");
}

Vector3f Warp::squareToUniformHemisphere(const Point2f &sample) {
    // Calculate the azimuthal and polar angles
    float phi = 2.0f * M_PI * sample.x();
    float theta = 0.5f * M_PI * sample.y(); 

    // Convert spherical coordinates to Cartesian coordinates (x, y, z)
    float x = std::sin(theta) * std::cos(phi);
    float y = std::sin(theta) * std::sin(phi);
    float z = std::cos(theta);

    // Return the 3D point on the unit sphere
    return Vector3f(x, y, z);
}

float Warp::squareToUniformHemispherePdf(const Vector3f &v) {
    throw NoriException("Warp::squareToUniformHemispherePdf() is not yet implemented!");
}

Vector3f Warp::squareToCosineHemisphere(const Point2f &sample) {
    throw NoriException("Warp::squareToCosineHemisphere() is not yet implemented!");
}

float Warp::squareToCosineHemispherePdf(const Vector3f &v) {
    throw NoriException("Warp::squareToCosineHemispherePdf() is not yet implemented!");
}

Vector3f Warp::squareToBeckmann(const Point2f &sample, float alpha) {
    throw NoriException("Warp::squareToBeckmann() is not yet implemented!");
}

float Warp::squareToBeckmannPdf(const Vector3f &m, float alpha) {
    throw NoriException("Warp::squareToBeckmannPdf() is not yet implemented!");
}

NORI_NAMESPACE_END
