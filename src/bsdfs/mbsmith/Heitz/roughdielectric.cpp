#include <memory>

#include <mitsuba/core/fresolver.h>
#include <mitsuba/hw/basicshader.h>
#include <mitsuba/render/bsdf.h>

#include "../../ior.h"
#include "../../microfacet.h"
#include "MicrosurfaceScattering.h"

MTS_NAMESPACE_BEGIN

class RoughDielectric : public BSDF {
public:
    RoughDielectric(const Properties &props) : BSDF(props)
    {
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

        Float intIOR = lookupIOR(props, "intIOR", "bk7");
        Float extIOR = lookupIOR(props, "extIOR", "air");
        m_eta        = intIOR / extIOR;
        m_invEta     = 1 / m_eta;

        dielectric_ptr =
                std::make_unique<MicrosurfaceDielectric>(height_uniform, slope_beckmann, m_alphaU, m_alphaV, m_eta);
    }

    RoughDielectric(Stream *stream, InstanceManager *manager) : BSDF(stream, manager) {}

    Spectrum eval(const BSDFSamplingRecord &bRec, EMeasure measure) const override
    {
        if (measure != ESolidAngle || Frame::cosTheta(bRec.wi) <= 0 || Frame::cosTheta(bRec.wo) <= 0 ||
            ((bRec.component != -1 && bRec.component != 0) || !(bRec.typeMask & EGlossyReflection)))
            return Spectrum(0.0f);
        glm::vec3 wi(bRec.wi.x, bRec.wi.y, bRec.wi.z);
        glm::vec3 wo(bRec.wo.x, bRec.wo.y, bRec.wo.z);
        return Spectrum(dielectric_ptr->eval(wi, wo));
    }

    Float pdf(const BSDFSamplingRecord &bRec, EMeasure measure) const override
    {
        if (measure != ESolidAngle)
            return 0.0f;

        /* Determine the type of interaction */
        bool hasReflection   = ((bRec.component == -1 || bRec.component == 0) && (bRec.typeMask & EGlossyReflection)),
             hasTransmission = ((bRec.component == -1 || bRec.component == 1) && (bRec.typeMask & EGlossyTransmission)),
             reflect         = Frame::cosTheta(bRec.wi) * Frame::cosTheta(bRec.wo) > 0;

        Vector wh;
        Float dwh_dwo;

        if (reflect)
        {
            /* Zero probability if this component was not requested */
            if ((bRec.component != -1 && bRec.component != 0) || !(bRec.typeMask & EGlossyReflection))
                return 0.0f;

            /* Calculate the reflection half-vector */
            wh = normalize(bRec.wo + bRec.wi);

            /* Jacobian of the half-direction mapping */
            dwh_dwo = 1.0f / (4.0f * dot(bRec.wo, wh));
        }
        else
        {
            /* Zero probability if this component was not requested */
            if ((bRec.component != -1 && bRec.component != 1) || !(bRec.typeMask & EGlossyTransmission))
                return 0.0f;

            /* Calculate the transmission half-vector */
            Float eta = Frame::cosTheta(bRec.wi) > 0 ? m_eta : m_invEta;

            wh = normalize(bRec.wi + bRec.wo * eta);

            /* Jacobian of the half-direction mapping */
            Float sqrtDenom = dot(bRec.wi, wh) + eta * dot(bRec.wo, wh);
            dwh_dwo         = (eta * eta * dot(bRec.wo, wh)) / (sqrtDenom * sqrtDenom);
        }

        /* Ensure that the half-vector points into the
        same hemisphere as the macrosurface normal */
        float alpha_x = m_alphaU;
        float alpha_y = m_alphaV;
        MicrofacetDistribution distr(m_type, alpha_x, alpha_y, true);

        wh *= math::signum(Frame::cosTheta(wh));
        //	Float s = math::signum(Frame::cosTheta(bRec.wi));
        //	float G1 = computeG1(s * bRec.wi, alpha_x, alpha_y);
        //	Float prob = std::max(0.0f, dot(wh, bRec.wi)) * distr.eval(wh) * G1 / Frame::cosTheta(bRec.wi);

        Float prob = distr.pdf(math::signum(Frame::cosTheta(bRec.wi)) * bRec.wi, wh);
        if (hasTransmission && hasReflection)
        {
            Float F = fresnelDielectricExt(dot(bRec.wi, wh), m_eta);
            prob *= reflect ? F : (1 - F);
        }

        // single-scattering PDF + diffuse
        // otherwise too many fireflies due to lack of multiple-scattering PDF
        // (MIS works even if the PDF is wrong and not normalized)
        return std::abs(prob /* * s*/ * dwh_dwo) + Frame::cosTheta(bRec.wo);
    }

    Spectrum sample(BSDFSamplingRecord &bRec, Float &pdf, const Point2 &sample) const override
    {
        if (Frame::cosTheta(bRec.wi) < 0 ||
            ((bRec.component != -1 && bRec.component != 0) || !(bRec.typeMask & EGlossyReflection)))
            return Spectrum(0.f);

        pdf = this->pdf(bRec, ESolidAngle);

        glm::vec3 wi(bRec.wi.x, bRec.wi.y, bRec.wi.z);
        auto res        = dielectric_ptr->sample(wi);
        glm::vec3 wo    = res.first;
        Spectrum weight = res.second;

        bRec.wo             = Vector3(wo.x, wo.y, wo.z);
        bool initialOutside = bRec.wi.z > 0;
        bool lastOutside    = bRec.wo.z > 0;
        if ((initialOutside && lastOutside) || (!initialOutside && !lastOutside))
        {
            bRec.sampledComponent = 0;
            bRec.sampledType      = EGlossyReflection;
            bRec.eta              = 1.f;

            if (Frame::cosTheta(bRec.wi) * Frame::cosTheta(bRec.wo) <= 0)
                return Spectrum(0.0f);
            return weight;
        }
        else
        {
            bRec.sampledComponent = 1;
            bRec.sampledType      = EGlossyTransmission;
            bRec.eta              = initialOutside ? m_eta : m_invEta;

            Float factor = (bRec.mode == ERadiance) ? (initialOutside ? m_invEta : m_eta) : 1.0f;
            return factor * factor * weight;
        }
    }
    Spectrum sample(BSDFSamplingRecord &bRec, const Point2 &sample) const override
    {
        glm::vec3 wi(bRec.wi.x, bRec.wi.y, bRec.wi.z);
        auto res        = dielectric_ptr->sample(wi);
        glm::vec3 wo    = res.first;
        Spectrum weight = res.second;

        bRec.wo = Vector3(wo.x, wo.y, wo.z);

        bool initialOutside = bRec.wi.z > 0;
        bool lastOutside    = bRec.wo.z > 0;
        if ((initialOutside && lastOutside) || (!initialOutside && !lastOutside))
        {
            bRec.sampledComponent = 0;
            bRec.sampledType      = EGlossyReflection;
            bRec.eta              = 1.f;

            if (Frame::cosTheta(bRec.wi) * Frame::cosTheta(bRec.wo) <= 0)
                return Spectrum(0.0f);
            return weight;
        }
        else
        {
            bRec.sampledComponent = 1;
            bRec.sampledType      = EGlossyTransmission;
            bRec.eta              = initialOutside ? m_eta : m_invEta;

            Float factor = (bRec.mode == ERadiance) ? (initialOutside ? m_invEta : m_eta) : 1.0f;
            return factor * factor * weight;
        }
    }

    MTS_DECLARE_CLASS()
private:
    std::unique_ptr<Microsurface> dielectric_ptr;
    MicrofacetDistribution::EType m_type;
    Float m_eta, m_invEta;
    Float m_alphaU, m_alphaV;
};

MTS_IMPLEMENT_CLASS_S(RoughDielectric, false, BSDF)
MTS_EXPORT_PLUGIN(RoughDielectric, "Rough dielectric BSDF");
MTS_NAMESPACE_END
