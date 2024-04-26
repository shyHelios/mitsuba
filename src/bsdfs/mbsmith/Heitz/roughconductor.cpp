#include <memory>

#include <mitsuba/core/fresolver.h>
#include <mitsuba/hw/basicshader.h>
#include <mitsuba/render/bsdf.h>

#include "../../ior.h"
#include "../../microfacet.h"
#include "MicrosurfaceScattering.h"

MTS_NAMESPACE_BEGIN

class RoughConductor : public BSDF {
public:
    RoughConductor(const Properties &props) : BSDF(props)
    {
        std::string materialName = props.getString("material", "none");
        bool height_uniform      = props.getBoolean("height_uniform", false);
        std::string distribution = props.getString("distribution", "ggx");
        bool slope_beckmann      = false;
        m_type                   = MicrofacetDistribution::EGGX;
        if (distribution == "beckmann")
        {
            slope_beckmann = true;
            m_type         = MicrofacetDistribution::EBeckmann;
        }
        m_alphaU = m_alphaV = props.getFloat("alpha", 0.3);
        conductor_ptr = std::make_unique<MicrosurfaceConductor>(materialName, height_uniform, slope_beckmann, m_alphaU,
                                                                m_alphaV);
    }

    RoughConductor(Stream *stream, InstanceManager *manager) : BSDF(stream, manager) {}

    Spectrum eval(const BSDFSamplingRecord &bRec, EMeasure measure) const override
    {
        if (measure != ESolidAngle || Frame::cosTheta(bRec.wi) <= 0 || Frame::cosTheta(bRec.wo) <= 0 ||
            ((bRec.component != -1 && bRec.component != 0) || !(bRec.typeMask & EGlossyReflection)))
            return Spectrum(0.0f);
        glm::vec3 wi(bRec.wi.x, bRec.wi.y, bRec.wi.z);
        glm::vec3 wo(bRec.wo.x, bRec.wo.y, bRec.wo.z);
        return Spectrum(conductor_ptr->eval(wi, wo));
    }

    Float pdf(const BSDFSamplingRecord &bRec, EMeasure measure) const override
    {
        if (measure != ESolidAngle || Frame::cosTheta(bRec.wi) <= 0 || Frame::cosTheta(bRec.wo) <= 0 ||
            ((bRec.component != -1 && bRec.component != 0) || !(bRec.typeMask & EGlossyReflection)))
            return 0.f;

        // single scattering pdf + diffuse
        return evalBouncePdf(bRec) + Frame::cosTheta(bRec.wo);
    }

    Spectrum sample(BSDFSamplingRecord &bRec, Float &pdf, const Point2 &sample) const override
    {
        if (Frame::cosTheta(bRec.wi) < 0 ||
            ((bRec.component != -1 && bRec.component != 0) || !(bRec.typeMask & EGlossyReflection)))
            return Spectrum(0.f);

        glm::vec3 wi(bRec.wi.x, bRec.wi.y, bRec.wi.z);

        auto res        = conductor_ptr->sample(wi);
        glm::vec3 wo    = res.first;
        Spectrum weight = res.second;

        bRec.wo               = Vector3(wo.x, wo.y, wo.z);
        bRec.eta              = 1.f;
        bRec.sampledComponent = 0;
        bRec.sampledType      = EGlossyReflection;

        pdf = this->pdf(bRec, ESolidAngle);

        /* Side check */
        if (Frame::cosTheta(bRec.wo) <= 0)
            return Spectrum(0.f);

        return weight;
    }
    Spectrum sample(BSDFSamplingRecord &bRec, const Point2 &sample) const override
    {
        glm::vec3 wi(bRec.wi.x, bRec.wi.y, bRec.wi.z);
        auto res        = conductor_ptr->sample(wi);
        glm::vec3 wo    = res.first;
        Spectrum weight = res.second;

        bRec.wo               = Vector3(wo.x, wo.y, wo.z);
        bRec.eta              = 1.f;
        bRec.sampledComponent = 0;
        bRec.sampledType      = EGlossyReflection;

        return weight;
    }

    MTS_DECLARE_CLASS()
private:
    /// 仅用来计算pdf
    float evalBouncePdf(const BSDFSamplingRecord &bRec) const
    {
        /* Calculate the reflection half-vector */
        Vector H = normalize(bRec.wo + bRec.wi);
        Float D  = conductor_ptr->m_microsurfaceslope->D(glm::vec3(H.x, H.y, H.z));
        float G1 = std::min(1.0f, conductor_ptr->G_1(glm::vec3(bRec.wi.x, bRec.wi.y, bRec.wi.z)));

        Float model = D * G1 / (4 * bRec.wi.z);
        return model;
    }

    std::unique_ptr<Microsurface> conductor_ptr;
    MicrofacetDistribution::EType m_type;
    Float m_alphaU, m_alphaV;
};

MTS_IMPLEMENT_CLASS_S(RoughConductor, false, BSDF)
MTS_EXPORT_PLUGIN(RoughConductor, "Rough conductor BRDF");
MTS_NAMESPACE_END
