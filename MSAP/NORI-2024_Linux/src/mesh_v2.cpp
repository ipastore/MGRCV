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

    m_pdf.reserve(m_F.cols());
    for (n_UINT i = 0; i < m_F.cols(); i++) //Depending on the number of triangles
    {
        float area = surfaceArea(i); //We get the area of the triangle
        m_pdf.append(area); // Append it to the list m_pdf 
    }
    m_pdf.normalize();  // this is done in order to sample the triangles with respect to their surface area
}

float Mesh::surfaceArea(n_UINT index) const {
    n_UINT i0 = m_F(0, index), i1 = m_F(1, index), i2 = m_F(2, index);

    const Point3f p0 = m_V.col(i0), p1 = m_V.col(i1), p2 = m_V.col(i2);

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

BoundingBox3f Mesh::getBoundingBox(n_UINT index) const {
    BoundingBox3f result(m_V.col(m_F(0, index)));
    result.expandBy(m_V.col(m_F(1, index)));
    result.expandBy(m_V.col(m_F(2, index)));
    return result;
}

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
void Mesh::samplePosition(const Point2f &sample, Point3f &p, Normal3f &n, Point2f &uv) const
{
	// throw NoriException("Mesh::samplePosition() is not yet implemented!");
    // get the triangle index, pdf is proportional to the surface area of the triangle
    Point2f randomSample = sample;
    size_t triangle_index = m_pdf.sampleReuse(randomSample.x());   // reuse the sample for the triangle index
    // now we have the triangle index, we can sample a point on the triangle
    // first, we get the vertices of the triangle
    n_UINT i0 = m_F(0, triangle_index), i1 = m_F(1, triangle_index), i2 = m_F(2, triangle_index);   // indices of the vertices
    const Point3f v0 = m_V.col(i0), v1 = m_V.col(i1), v2 = m_V.col(i2);   // vertices of the triangle
    // get the baricentric coordinates of the sample
    Point2f bar_coord = Warp::squareToUniformTriangle(sample);
    float u = bar_coord.x(), v = bar_coord.y(), w = 1.0f - u - v;
    // interpolate those coordinates to the triangle via the vertices
    p = v0 * u + v1 * v + v2 * w;

    // now do the same for the normal
    // initialize the normals
    
    // FIXME: this breaks table, uncomment for working version (only to be used with table)
    // check if the mesh has normals
    if (m_N.size() > 0) {
        // n0 = m_N.col(m_F(0, triangle_index));
        // n1 = m_N.col(m_F(1, triangle_index));
        // n2 = m_N.col(m_F(2, triangle_index));
        const Normal3f n0 = m_N.col(i0), n1 = m_N.col(i1), n2 = m_N.col(i2);
        n = n0 * u + n1 * v + n2 * w;
    }else{  // if the mesh does not have normals, compute them
        // n0 = (v1 - v0).cross(v2 - v0);
        // n1 = (v2 - v1).cross(v0 - v1);
        // n2 = (v0 - v2).cross(v1 - v2);
        const Vector3f edge1 = v1 - v0, edge2 = v2 - v0;
        n = (edge1.cross(edge2)).normalized();
    }
    
    // now do the same for the uv coordinates
    Point2f uv0 = m_UV.col(i0);
    Point2f uv1 = m_UV.col(i1);
    Point2f uv2 = m_UV.col(i2);
    // interpolate those coordinates to the triangle via the vertices
    uv = u * uv0 + v * uv1 + w * uv2;
} 

/// Return the surface area of the given triangle
float Mesh::pdf(const Point3f &p) const
{
	// throw NoriException("Mesh::pdf() is not yet implemented!");	
	return m_pdf.getNormalization();
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
        "  emitter = %s,\n"
        "  medium = %s\n"
        "]",
        m_name,
        m_V.cols(),
        m_F.cols(),
        m_bsdf ? indent(m_bsdf->toString()) : std::string("null"),
        m_emitter ? indent(m_emitter->toString()) : std::string("null")
        // m_medium ? indent(m_medium->toString()) : std::string("null")
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
        "  mesh = %s,\n"
        "]",
        p.toString(),
        t,
        uv.toString(),
        indent(shFrame.toString()),
        indent(geoFrame.toString()),
        mesh ? mesh->toString() : std::string("null")
        // medium ? medium->toString() : std::string("null")
    );
}

NORI_NAMESPACE_END
