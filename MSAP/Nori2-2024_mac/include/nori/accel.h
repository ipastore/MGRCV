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

#pragma once

#include <nori/mesh.h>

NORI_NAMESPACE_BEGIN

/**
 * \brief Acceleration data structure for ray intersection queries
 *
 * The current implementation falls back to a brute force loop
 * through the geometry.
 */
class Accel {
	friend class BVHBuildTask;
public:
	/// Create a new and empty BVH
	Accel() { m_meshOffset.push_back(0u); }

	/// Release all resources
	virtual ~Accel() { clear(); };

	/// Release all resources
	void clear();

	/**
	 * \brief Register a triangle mesh for inclusion in the BVH.
	 *
	 * This function can only be used before \ref build() is called
	 */
	void addMesh(Mesh *mesh);

	/// Build the BVH
	void build();

	/**
	 * \brief Intersect a ray against all triangle meshes registered
	 * with the BVH
	 *
	 * Detailed information about the intersection, if any, will be
	 * stored in the provided \ref Intersection data record.
	 *
	 * The <tt>shadowRay</tt> parameter specifies whether this detailed
	 * information is really needed. When set to \c true, the
	 * function just checks whether or not there is occlusion, but without
	 * providing any more detail (i.e. \c its will not be filled with
	 * contents). This is usually much faster.
	 *
	 * \return \c true If an intersection was found
	 */
	bool rayIntersect(const Ray3f &ray, Intersection &its,
		bool shadowRay = false) const;

	/// Return the total number of meshes registered with the BVH
	n_UINT getMeshCount() const { return (n_UINT)m_meshes.size(); }

	/// Return the total number of internally represented triangles 
	n_UINT getTriangleCount() const { return m_meshOffset.back(); }

	/// Return one of the registered meshes
	Mesh *getMesh(n_UINT idx) { return m_meshes[idx]; }

	/// Return one of the registered meshes (const version)
	const Mesh *getMesh(n_UINT idx) const { return m_meshes[idx]; }

	//// Return an axis-aligned bounding box containing the entire tree
	const BoundingBox3f &getBoundingBox() const {
		return m_bbox;
	}

protected:
	/**
	 * \brief Compute the mesh and triangle indices corresponding to
	 * a primitive index used by the underlying generic BVH implementation.
	 */
	n_UINT findMesh(n_UINT &idx) const {
		auto it = std::lower_bound(m_meshOffset.begin(), m_meshOffset.end(), idx + 1) - 1;
		idx -= *it;
		return (n_UINT)(it - m_meshOffset.begin());
	}

	//// Return an axis-aligned bounding box containing the given triangle
	BoundingBox3f getBoundingBox(n_UINT index) const {
		n_UINT meshIdx = findMesh(index);
		return m_meshes[meshIdx]->getBoundingBox(index);
	}

	//// Return the centroid of the given triangle
	Point3f getCentroid(n_UINT index) const {
		n_UINT meshIdx = findMesh(index);
		return m_meshes[meshIdx]->getCentroid(index);
	}

	/// Compute internal tree statistics
	std::pair<float, n_UINT> statistics(n_UINT index = 0) const;

	/* BVH node in 32 bytes */
	struct BVHNode {
		union {
			struct {
				unsigned flag : 1;
				uint32_t size : 31;
				n_UINT start;
			} leaf;

			struct {
				unsigned flag : 1;
				uint32_t axis : 31;
				n_UINT rightChild;
			} inner;

			uint64_t data;
		};
		BoundingBox3f bbox;

		bool isLeaf() const {
			return leaf.flag == 1;
		}

		bool isInner() const {
			return leaf.flag == 0;
		}

		bool isUnused() const {
			return data == 0;
		}

		n_UINT start() const {
			return leaf.start;
		}

		n_UINT end() const {
			return leaf.start + leaf.size;
		}
	};
private:
	std::vector<Mesh *> m_meshes;       ///< List of meshes registered with the BVH
	std::vector<n_UINT> m_meshOffset; ///< Index of the first triangle for each shape
	std::vector<BVHNode> m_nodes;       ///< BVH nodes
	std::vector<n_UINT> m_indices;    ///< Index references by BVH nodes
	BoundingBox3f m_bbox;               ///< Bounding box of the entire BVH
};


NORI_NAMESPACE_END
