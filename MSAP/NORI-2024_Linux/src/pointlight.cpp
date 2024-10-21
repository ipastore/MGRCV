#include <nori/emitter.h>

NORI_NAMESPACE_BEGIN

// Define the PointEmitter class, which inherits from the Emitter class
class PointEmitter : public Emitter {
public:
    // Constructor to initialize the emitter with properties
    PointEmitter(const PropertyList &props) {
        // Set the type of the emitter to a point emitter
        m_type = EmitterType::EMITTER_POINT;

        // Initialize the position of the emitter, defaulting to (0, 100, 0)
        m_position = props.getPoint("position", Point3f(0., 100., 0.));

        // Initialize the radiance (color) of the emitter, defaulting to white (1.0)
        m_radiance = props.getColor("radiance", Color3f(1.f));
    }

    // Function to return a string representation of the emitter
    virtual std::string toString() const {
        return tfm::format(
            "PointEmitter[\n"
            " position = %s,\n"
            " radiance = %s,\n"
            "]",
            m_position.toString(),
            m_radiance.toString()
        );
    }

    // Function to evaluate the emitter, but returns 0 as point emitters are not directly sampled
    virtual Color3f eval(const EmitterQueryRecord &lRec) const {
        // Since randomly sampling a point has probability 0, return 0
        // This function assumes that a ray have been traced towards
        // the light source. However, since the probability of randomly
        // sampling a point in space is 0, its evaluation returns 0.
        return 0.;
    }

    // Function to sample the light source
    virtual Color3f sample(EmitterQueryRecord &lRec,
                           const Point2f &sample,
                           float optional_u) const {
        // Set the sample position to the emitter's position
        lRec.p = m_position;

        // Compute the distance between the reference point and the emitter
        lRec.dist = (lRec.p - lRec.ref).norm();

        // Calculate the direction from the reference point to the emitter
        lRec.wi = (lRec.p - lRec.ref) / lRec.dist;

        // Set the probability density function (pdf) to 1 (though technically should be infinite)
        lRec.pdf = 1.;

        // Return the radiance divided by the square of the distance (falloff with distance)
        return m_radiance / (lRec.dist * lRec.dist);
    }

    // Function to return the pdf, which is always 1 for point emitters
    virtual float pdf(const EmitterQueryRecord &lRec) const {
        return 1.;
    }

protected:
    // Position of the point emitter
    Point3f m_position;

    // Radiance (or color) of the emitter
    Color3f m_radiance;
};

NORI_REGISTER_CLASS(PointEmitter, "pointlight")
NORI_NAMESPACE_END