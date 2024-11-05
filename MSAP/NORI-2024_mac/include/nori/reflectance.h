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

#pragma once

#include <nori/common.h>
#include <nori/color.h>

NORI_NAMESPACE_BEGIN

namespace Reflectance {
	
	/**
	 * \brief Computes the refracted vector following Sahl-Snell's law
	 *
	 * \param wi
	 *      Incident vector 
	 * \param n
	 *      Surface normal 
	 * \param extIOR
	 *      Refractive index of the side that contains the surface normal
	 * \param intIOR
	 *      Refractive index of the interior
	 */
	extern Vector3f refract(const Vector3f& wi, const Vector3f& n, float extIOR, float intIOR);

	/**
	 * \brief Calculates the unpolarized fresnel reflection coefficient for a
	 * dielectric material. Handles incidence from either side (i.e.
	 * \code cosThetaI<0 is allowed).
	 *
	 * \param cosThetaI
	 *      Cosine of the angle between the normal and the incident ray
	 * \param extIOR
	 *      Refractive index of the side that contains the surface normal
	 * \param intIOR
	 *      Refractive index of the interior
	 */
	extern float fresnel(float cosThetaI, float extIOR, float intIOR);

	/**
	 * \brief Calculates the fresnel reflection coefficient based on the
	 * Schlick's approximation.
	 *
	 * \param cosThetaI
	 *      Cosine of the angle between the normal and the incident ray
	 * \param extIOR
	 *      Refractive index of the side that contains the surface normal
	 * \param intIOR
	 *      Refractive index of the interior
	 */
	extern Color3f fresnel(float cosThetaI, const Color3f& R0);

	/**
	 * \brief Calculates the Smith's geometry term
	 *
	 * \param alpha
	 *      The roughness of the surface
	 * \param wv
	 *      The viewing direction
	 * \param wh
	 *      The half vector
	 */
	extern float G1(const Vector3f& wv, const Vector3f &wh, float alpha);

	/**
	 * \brief Calculates the Beckmann's NDF D(wh). Note that it is different
	 *		 from Warp::BeckmannPDF, which is D(wh)cos(thetah)
	 *
	 * \param alpha
	 *      The roughness of the surface
	 * \param wh
	 *      The half vector
	 */
	extern float BeckmannNDF(const Vector3f& wh, float alpha);

};

NORI_NAMESPACE_END