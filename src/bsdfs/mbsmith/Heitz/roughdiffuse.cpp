#include <memory>

#include <mitsuba/core/fresolver.h>
#include <mitsuba/hw/basicshader.h>
#include <mitsuba/render/bsdf.h>

#include "../../ior.h"
#include "../../microfacet.h"
#include "MicrosurfaceScattering.h"

MTS_NAMESPACE_BEGIN

class RoughDiffuse : public BSDF {
public:
    RoughDiffuse(const Properties &props) : BSDF(props)
    {
        bool height_uniform      = props.getBoolean("height_uniform", false);
        std::string distribution = props.getString("distribution", "ggx");
        bool slope_beckmann      = false;
        if (distribution == "beckmann")
            slope_beckmann = true;
        m_alphaU = m_alphaV = props.getFloat("alpha", 0.3);
        diffuse_ptr = std::make_unique<MicrosurfaceDiffuse>(height_uniform, slope_beckmann, m_alphaU,
                                                                m_alphaV);
    }

    RoughDiffuse(Stream *stream, InstanceManager *manager) : BSDF(stream, manager) {}

    Spectrum eval(const BSDFSamplingRecord &bRec, EMeasure measure) const override
    {
        if (measure != ESolidAngle || Frame::cosTheta(bRec.wi) <= 0 || Frame::cosTheta(bRec.wo) <= 0 ||
            ((bRec.component != -1 && bRec.component != 0) || !(bRec.typeMask & EGlossyReflection)))
            return Spectrum(0.0f);
        glm::vec3 wi(bRec.wi.x, bRec.wi.y, bRec.wi.z);
        glm::vec3 wo(bRec.wo.x, bRec.wo.y, bRec.wo.z);
        return Spectrum(diffuse_ptr->eval(wi, wo));
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

        auto res        = diffuse_ptr->sample(wi);
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
        auto res        = diffuse_ptr->sample(wi);
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
        Float D  = diffuse_ptr->m_microsurfaceslope->D(glm::vec3(H.x, H.y, H.z));
        float G1 = std::min(1.0f, diffuse_ptr->G_1(glm::vec3(bRec.wi.x, bRec.wi.y, bRec.wi.z)));

        Float model = D * G1 / (4 * bRec.wi.z);
        return model;
    }

    std::unique_ptr<Microsurface> diffuse_ptr;
    Float m_alphaU, m_alphaV;
};

MTS_IMPLEMENT_CLASS_S(RoughDiffuse, false, BSDF)
MTS_EXPORT_PLUGIN(RoughDiffuse, "Rough diffuse BRDF");
MTS_NAMESPACE_END
