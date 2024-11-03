/*
    This file is part of Nori, a simple educational ray tracer

    Copyright (c) 2015 by Wenzel Jakob

    v1 - Dec 2020
    Copyright (c) 2020 by Adrian Jarabo

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


#include <nori/mesh.h>
#include <nori/bbox.h>
#include <nori/bsdf.h>
#include <nori/emitter.h>
#include <nori/warp.h>
#include <Eigen/Geometry>

NORI_NAMESPACE_BEGIN

Mesh::Mesh() { }

Mesh::~Mesh() {
    m_pdf.clear();
    delete m_bsdf;
    delete m_emitter;
}

void Mesh::activate() {
    if (!m_bsdf) {
        /* If no material was assigned, instantiate a diffuse BRDF */
        m_bsdf = static_cast<BSDF *>(
            NoriObjectFactory::createInstance("diffuse", PropertyList()));
    }

    // Inicializamos la m_pdf para reservar espacio para cada una de las superficies (triangle) de m_F
    m_pdf.reserve(m_F.cols());
    for (n_UINT i = 0; i < m_F.cols(); ++i) {
        float area = surfaceArea(i);  // Compute area of each triangle
        m_pdf.append(area);            
    }
    m_pdf.normalize();
}

/// Return the surface area of the given triangle
float Mesh::surfaceArea(n_UINT index) const {
    // Get the indices of the vertices of the triangle
    n_UINT i0 = m_F(0, index), i1 = m_F(1, index), i2 = m_F(2, index);

    // Get the vertices of the triangle
    const Point3f p0 = m_V.col(i0), p1 = m_V.col(i1), p2 = m_V.col(i2);

    // Compute the area of the triangle
    return 0.5f * Vector3f((p1 - p0).cross(p2 - p0)).norm();
}

bool Mesh::rayIntersect(n_UINT index, const Ray3f &ray, float &u, float &v, float &t) const {
    n_UINT i0 = m_F(0, index), i1 = m_F(1, index), i2 = m_F(2, index);
    const Point3f p0 = m_V.col(i0), p1 = m_V.col(i1), p2 = m_V.col(i2);

    /* Find vectors for two edges sharing v[0] */
    Vector3f edge1 = p1 - p0, edge2 = p2 - p0;

    /* Begin calculating determinant - also used to calculate U parameter */
    Vector3f pvec = ray.d.cross(edge2);

    /* If determinant is near zero, ray lies in plane of triangle */
    float det = edge1.dot(pvec);

    if (det > -1e-8f && det < 1e-8f)
        return false;
    float inv_det = 1.0f / det;

    /* Calculate distance from v[0] to ray origin */
    Vector3f tvec = ray.o - p0;

    /* Calculate U parameter and test bounds */
    u = tvec.dot(pvec) * inv_det;
    if (u < 0.0 || u > 1.0)
        return false;

    /* Prepare to test V parameter */
    Vector3f qvec = tvec.cross(edge1);

    /* Calculate V parameter and test bounds */
    v = ray.d.dot(qvec) * inv_det;
    if (v < 0.0 || u + v > 1.0)
        return false;

    /* Ray intersects triangle -> compute t */
    t = edge2.dot(qvec) * inv_det;

    return t >= ray.mint && t <= ray.maxt;
}

// Computes the bounding box of a single triangle in the mesh.
BoundingBox3f Mesh::getBoundingBox(n_UINT index) const {
    // Get the indices of the vertices of the triangle
    // Get the vertices of the triangle
    // Compute the bounding box of the triangle
    BoundingBox3f result(m_V.col(m_F(0, index)));
    result.expandBy(m_V.col(m_F(1, index)));
    result.expandBy(m_V.col(m_F(2, index)));
    return result;
}

//// Return the centroid of the given triangle
Point3f Mesh::getCentroid(n_UINT index) const {
    return (1.0f / 3.0f) *
        (m_V.col(m_F(0, index)) +
         m_V.col(m_F(1, index)) +
         m_V.col(m_F(2, index)));
}

/**
 * \brief Uniformly sample a position on the mesh with
 * respect to surface area. Returns both position and normal
 */
void Mesh::samplePosition(const Point2f &sample, Point3f &p, Normal3f &n, Point2f &uv) const {
    // Paso 1: Seleccionar el triángulo

    float sample_x = sample.x();
    float sample_y = sample.y();
    size_t triangleIndex = m_pdf.sampleReuse(sample_x);


    n_UINT i0 = m_F(0, triangleIndex);  // Índice del primer vértice
    n_UINT i1 = m_F(1, triangleIndex);  // Índice del segundo vértice
    n_UINT i2 = m_F(2, triangleIndex);  // Índice del tercer vértice

    // Obtener las posiciones de los vértices del triángulo
    const Point3f &v0 = m_V.col(i0);
    const Point3f &v1 = m_V.col(i1);
    const Point3f &v2 = m_V.col(i2);

    // Paso 2: Generamos las coordenadas baricéntricas uniformes dentro del triángulo
    Point2f barycentric = Warp::squareToUniformTriangle(Point2f(sample_x, sample_y)); // Reuse sample.y() for the barycentric coordinates
    float b0 = 1.0f - barycentric.x() - barycentric.y(); // Coordenada baricéntrica para v0
    float b1 = barycentric.x();                          // Coordenada baricéntrica para v1
    float b2 = barycentric.y();                          // Coordenada baricéntrica para v2

    // Paso 3: Interpolamos la posición en el triángulo usando las coordenadas baricéntricas
    p = b0 * v0 + b1 * v1 + b2 * v2;

    // Paso 4: Interpolamos la normal (si las normales están definidas)
    if (m_N.size() > 0) {  // Verificamos si existen normales
        const Normal3f &n0 = m_N.col(i0);
        const Normal3f &n1 = m_N.col(i1);
        const Normal3f &n2 = m_N.col(i2);
        n = (b0 * n0 + b1 * n1 + b2 * n2).normalized();
    } else {
        // Si no hay normales, calcular la normal del triángulo usando el producto cruzado
        n = (v1 - v0).cross(v2 - v0).normalized();
    }

    // Paso 5: Interpolamoa las coordenadas UV (si existen)
    if (m_UV.size() > 0) {  // Verificamos si existen coordenadas UV
        const Point2f &uv0 = m_UV.col(i0);
        const Point2f &uv1 = m_UV.col(i1);
        const Point2f &uv2 = m_UV.col(i2);
        uv = b0 * uv0 + b1 * uv1 + b2 * uv2;
    } else {
        // Si no existen coordenadas UV, asignamos un valor predeterminado
        uv = Point2f(0.0f, 0.0f);
    }

}

/// Return the surface area of the given triangle
float Mesh::pdf(const Point3f &p) const {
    float totalArea = m_pdf.getNormalization();
    return (totalArea > 0.0f) ? (totalArea) : 0.0f;
    
}


void Mesh::addChild(NoriObject *obj, const std::string& name) {
    switch (obj->getClassType()) {
        case EBSDF:
            if (m_bsdf)
                throw NoriException(
                    "Mesh: tried to register multiple BSDF instances!");
            m_bsdf = static_cast<BSDF *>(obj);
            break;

        case EEmitter: {
                Emitter *emitter = static_cast<Emitter *>(obj);
                if (m_emitter)
                    throw NoriException(
                        "Mesh: tried to register multiple Emitter instances!");
                m_emitter = emitter;
            }
            break;

        default:
            throw NoriException("Mesh::addChild(<%s>) is not supported!",
                                classTypeName(obj->getClassType()));
    }
}

std::string Mesh::toString() const {
    return tfm::format(
        "Mesh[\n"
        "  name = \"%s\",\n"
        "  vertexCount = %i,\n"
        "  triangleCount = %i,\n"
        "  bsdf = %s,\n"
        "  emitter = %s\n"
        "]",
        m_name,
        m_V.cols(),
        m_F.cols(),
        m_bsdf ? indent(m_bsdf->toString()) : std::string("null"),
        m_emitter ? indent(m_emitter->toString()) : std::string("null")
    );
}

std::string Intersection::toString() const {
    if (!mesh)
        return "Intersection[invalid]";

    return tfm::format(
        "Intersection[\n"
        "  p = %s,\n"
        "  t = %f,\n"
        "  uv = %s,\n"
        "  shFrame = %s,\n"
        "  geoFrame = %s,\n"
        "  mesh = %s\n"
        "]",
        p.toString(),
        t,
        uv.toString(),
        indent(shFrame.toString()),
        indent(geoFrame.toString()),
        mesh ? mesh->toString() : std::string("null")
    );
}

NORI_NAMESPACE_END
