/*
    This file is an extension of Nori, a simple educational ray tracer
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

#pragma once

#include <nori/object.h>

NORI_NAMESPACE_BEGIN

/**
 * \brief Superclass of all textures to be used in Nori
 */
class Texture : public NoriObject {

public:
    /**
     * \brief Evaluate the texture for a 2D position
     *
     * \param uv
     *     The uv coordinates in the texture
     * \return
     *     The color of the texture
     */
    virtual Color3f eval(const Point2f& uv) const = 0;

    /**
     * \brief Return the type of object (i.e. Mesh/BSDF/etc.)
    * provided by this instance
    * */
    EClassType getClassType() const { return ETexture; }

};


class ConstantSpectrumTexture : public Texture {
public:
    ConstantSpectrumTexture(const PropertyList& props)
    {
        m_color = props.getColor("color", Color3f(0.5f));
    }

    ConstantSpectrumTexture(const Color3f& color): m_color(color){  }

    Color3f eval(const Point2f& uv) const { return m_color; }

    virtual std::string toString() const {
        return tfm::format("%s", m_color.toString());
    }
private:
    Color3f m_color;
};


NORI_NAMESPACE_END
