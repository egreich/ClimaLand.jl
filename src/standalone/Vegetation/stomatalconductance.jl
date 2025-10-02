export MedlynConductanceParameters,
    MedlynConductanceModel, PModelConductanceParameters, PModelConductance, OWUSConductanceParameters, OWUSConductanceModel
export OWUSCWDStaticParameters, OWUSConductanceCWDStatic

abstract type AbstractStomatalConductanceModel{FT} <:
              AbstractCanopyComponent{FT} end

"""x
    MedlynConductanceParameters{FT <: AbstractFloat}

The required parameters for the Medlyn stomatal conductance model.
$(DocStringExtensions.FIELDS)
"""
Base.@kwdef struct MedlynConductanceParameters{
    FT <: AbstractFloat,
    G1 <: Union{FT, ClimaCore.Fields.Field},
}
    "Relative diffusivity of water vapor (unitless)"
    Drel::FT
    "Minimum stomatal conductance mol/m^2/s"
    g0::FT
    "Slope parameter, inversely proportional to the square root of marginal water use efficiency (Pa^{1/2})"
    g1::G1
end

Base.eltype(::MedlynConductanceParameters{FT}) where {FT} = FT

struct MedlynConductanceModel{FT, MCP <: MedlynConductanceParameters{FT}} <:
       AbstractStomatalConductanceModel{FT}
    parameters::MCP
end

function MedlynConductanceModel{FT}(
    parameters::MedlynConductanceParameters{FT},
) where {FT <: AbstractFloat}
    return MedlynConductanceModel{eltype(parameters), typeof(parameters)}(
        parameters,
    )
end

ClimaLand.name(model::AbstractStomatalConductanceModel) = :conductance

ClimaLand.auxiliary_vars(model::MedlynConductanceModel) = (:r_stomata_canopy,)
ClimaLand.auxiliary_types(model::MedlynConductanceModel{FT}) where {FT} = (FT,)
ClimaLand.auxiliary_domain_names(::MedlynConductanceModel) = (:surface,)

"""
    update_canopy_conductance!(p, Y, model::MedlynConductanceModel, canopy)

Computes and updates the canopy-level conductance (units of m/s) according to the Medlyn model.

The moisture stress factor is applied to `An_leaf` already.
"""
function update_canopy_conductance!(p, Y, model::MedlynConductanceModel, canopy)
    c_co2_air = p.drivers.c_co2
    P_air = p.drivers.P
    T_air = p.drivers.T
    q_air = p.drivers.q
    earth_param_set = canopy.parameters.earth_param_set
    thermo_params = earth_param_set.thermo_params
    (; g1, g0, Drel) = canopy.conductance.parameters
    area_index = p.canopy.biomass.area_index
    LAI = area_index.leaf
    An_leaf = get_An_leaf(p, canopy.photosynthesis)
    R = LP.gas_constant(earth_param_set)
    FT = typeof(R)
    medlyn_factor = @. lazy(medlyn_term(g1, T_air, P_air, q_air, thermo_params))
    @. p.canopy.conductance.r_stomata_canopy =
        1 / (
            conductance_molar_flux_to_m_per_s(
                medlyn_conductance(g0, Drel, medlyn_factor, An_leaf, c_co2_air), #conductance, leaf level
                T_air,
                R,
                P_air,
            ) * max(LAI, sqrt(eps(FT)))
        ) # multiply by LAI treating all leaves as if they are in parallel
end

# For interfacing with ClimaParams

"""
    function MedlynConductanceParameters(
        toml_dict::CP.ParamDict;
        g1,
        g0 = toml_dict["min_stomatal_conductance"],
    )

TOML dict based constructor supplying default values for the
`MedlynConductanceParameters` struct.
"""
function MedlynConductanceParameters(
    toml_dict::CP.ParamDict;
    g1,
    g0 = toml_dict["min_stomatal_conductance"],
)
    name_map = (; :relative_diffusivity_of_water_vapor => :Drel,)

    parameters = CP.get_parameter_values(toml_dict, name_map, "Land")
    FT = CP.float_type(toml_dict)
    g1 = FT.(g1)
    G1 = typeof(g1)
    return MedlynConductanceParameters{FT, G1}(; g0, g1, parameters...)
end


#################### P model conductance ####################
"""
    PModelConductanceParameters{FT <: AbstractFloat}

The required parameters for the P-Model stomatal conductance model.
$(DocStringExtensions.FIELDS)
"""
Base.@kwdef struct PModelConductanceParameters{FT <: AbstractFloat}
    "Relative diffusivity of water vapor (unitless)"
    Drel::FT
end

Base.eltype(::PModelConductanceParameters{FT}) where {FT} = FT

struct PModelConductance{FT, PMCP <: PModelConductanceParameters{FT}} <:
       AbstractStomatalConductanceModel{FT}
    parameters::PMCP
end

function PModelConductance{FT}(
    parameters::PModelConductanceParameters{FT},
) where {FT <: AbstractFloat}
    return PModelConductance{eltype(parameters), typeof(parameters)}(parameters)
end

ClimaLand.auxiliary_vars(model::PModelConductance) = (:r_stomata_canopy,)
ClimaLand.auxiliary_types(model::PModelConductance{FT}) where {FT} = (FT,)
ClimaLand.auxiliary_domain_names(::PModelConductance) = (:surface,)

"""
    update_canopy_conductance!(p, Y, model::PModelConductance, canopy)

Computes and updates the canopy-level conductance (units of m/s) according to the P model. 
The P-model predicts the ratio of plant internal to external CO2 concentration χ, and therefore
the stomatal conductance can be inferred from their difference and the net assimilation rate `An`. 

Note that the moisture stress factor `βm` is applied to `An` already, so it is not applied again here. 
"""
function update_canopy_conductance!(p, Y, model::PModelConductance, canopy)
    c_co2_air = p.drivers.c_co2
    P_air = p.drivers.P
    T_air = p.drivers.T
    earth_param_set = canopy.parameters.earth_param_set
    (; Drel) = canopy.conductance.parameters
    area_index = p.canopy.biomass.area_index
    LAI = area_index.leaf
    ci = p.canopy.photosynthesis.ci             # internal CO2 partial pressure, Pa 
    An_canopy = p.canopy.photosynthesis.An          # net assimilation rate, mol m^-2 s^-1, canopy level
    R = LP.gas_constant(earth_param_set)
    FT = eltype(model.parameters)

    χ = @. lazy(ci / (c_co2_air * P_air))       # ratio of intercellular to ambient CO2 concentration, unitless
    @. p.canopy.conductance.r_stomata_canopy =
        1 / (
            conductance_molar_flux_to_m_per_s(
                gs_h2o_pmodel(χ, c_co2_air, An_canopy, Drel), # canopy level conductance in mol H2O/m^2/s
                T_air,
                R,
                P_air,
            ) + eps(FT)
        ) # avoids division by zero, since conductance is zero when An is zero 
end

#################### OWUS conductance (existing static back-compat) ####################
Base.@kwdef struct OWUSConductanceParameters{FT <: AbstractFloat}
    "Well-watered transpiration fraction (unitless), plateau of β(s)"
    fww::FT
    "Soil saturation threshold where down-regulation begins (unitless)"
    s_star::FT
    "Soil saturation at shutdown (unitless)"
    s_w::FT
    "Optional cap on stomatal conductance (mol m^-2 s^-1); Inf for none"
    gsw_max::FT = FT(Inf)
end
Base.eltype(::OWUSConductanceParameters{FT}) where {FT} = FT

struct OWUSConductanceModel{FT, OCP <: OWUSConductanceParameters{FT}} <:
       AbstractStomatalConductanceModel{FT}
    parameters::OCP
end

function OWUSConductanceModel{FT}(
    parameters::OWUSConductanceParameters{FT},
) where {FT <: AbstractFloat}
    return OWUSConductanceModel{eltype(parameters), typeof(parameters)}(parameters)
end

function OWUSConductanceModel{FT}(;
    canopy_params, soil_params, root_params, met_params=nothing, gsw_max = FT(Inf)
) where {FT <: AbstractFloat}
    owus = OWUSStomata.build_owus_from_ClimaLand(;
        canopy_params = canopy_params,
        soil_params   = soil_params,
        root_params   = root_params,
        met_params    = met_params,
        overrides     = NamedTuple()
    )
    pars = OWUSConductanceParameters{FT}(;
        fww     = FT(owus.fww),
        s_star  = FT(owus.s_star),
        s_w     = FT(owus.s_w),
        gsw_max = FT(owus.gsw_max),
    )
    return OWUSConductanceModel{FT}(pars)
end


ClimaLand.auxiliary_vars(::OWUSConductanceModel) = (:r_stomata_canopy, :gsw_leaf, :gsw_canopy)
ClimaLand.auxiliary_types(::OWUSConductanceModel{FT}) where {FT} = (FT, FT, FT)
ClimaLand.auxiliary_domain_names(::OWUSConductanceModel) = (:surface, :surface, :surface)

# --- soft getters ---
@inline _has(x, f::Symbol) = Base.hasproperty(x, f)
@inline _get(x, f::Symbol, dflt=nothing) = _has(x, f) ? getproperty(x,f) : dflt

# Soil saturation s \in [0,1].
# Field-valued soil saturation s ∈ [0,1]
@inline function _saturation_field(p, ::Any, ::Type{FT}) where {FT}
    # pick any canopy Field to clone shape/grid from
    like = p.drivers.T

    # 1) Prefer canopy soil_moisture_stress if it exposes θ info
    if Base.hasproperty(p.canopy, :soil_moisture_stress)
        sms = p.canopy.soil_moisture_stress

        if Base.hasproperty(sms, :θ) &&
           Base.hasproperty(sms, :θ_high) &&
           Base.hasproperty(sms, :θ_low)
            θ    = sms.θ
            θ_hi = sms.θ_high
            θ_lo = sms.θ_low
            return @. clamp((θ - θ_lo) / max(θ_hi - θ_lo, eps(FT)), FT(0), FT(1))
        elseif Base.hasproperty(sms, :s)
            s = sms.s
            return @. clamp(FT(s), FT(0), FT(1))
        elseif Base.hasproperty(sms, :βm)
            βm = sms.βm
            # treat βm as a saturation-like 0..1 proxy
            return @. clamp(FT(βm), FT(0), FT(1))
        end
    end

    # 2) Otherwise, try soil state with θ_r, ν
    if Base.hasproperty(p, :soil)
        soil = p.soil
        if Base.hasproperty(soil, :θ) && Base.hasproperty(soil, :parameters)
            θ = soil.θ
            pars = soil.parameters
            if Base.hasproperty(pars, :ν) && Base.hasproperty(pars, :θ_r)
                ν   = pars.ν
                θ_r = pars.θ_r
                return @. clamp((θ - θ_r) / max(ν - θ_r, eps(FT)), FT(0), FT(1))
            end
        end
    end
end

@inline function _E0_field(p, ::Any, ::Type{FT}) where {FT}
    if Base.hasproperty(p.canopy, :energy) && Base.hasproperty(p.canopy.energy, :E0)
        p.canopy.energy.E0
    elseif Base.hasproperty(p.drivers, :E0)
        p.drivers.E0
    else
        # fill a Field with a small constant default
        LAI = p.canopy.biomass.area_index.leaf
        @. zero(LAI) + FT(2.5e-3/86400)
    end
end

@inline function _VPD_field(p, ::Type{FT}) where {FT}
    T = p.drivers.T
    P = p.drivers.P
    q = p.drivers.q
    @inline _svp_pa(Tk) = FT(610.94) * exp((FT(17.625) * (Tk - FT(273.15))) / (Tk - FT(273.15) + FT(243.04)))
    e  = @. (q * P) / (FT(0.622) + (FT(1) - FT(0.622)) * q)
    es = @. _svp_pa(T)
    @. max(es - e, FT(0))
end

"""
    update_canopy_conductance!(p, Y, model::OWUSConductanceModel, canopy)

Static OWUS (back-compat, no time variability in parameters).
"""
function update_canopy_conductance!(p, Y, model::OWUSConductanceModel, canopy)
    FT   = eltype(model.parameters)
    pars = model.parameters
    earth = canopy.parameters.earth_param_set
    Rgas  = LP.gas_constant(earth)  # FT scalar

    # Inputs as Fields
    P_air = p.drivers.P
    T_air = p.drivers.T
    LAI   = p.canopy.biomass.area_index.leaf
    s     = _saturation_field(p, canopy, FT)
    E0    = _E0_field(p, canopy, FT)
    VPD   = _VPD_field(p, FT)

    # OWUS β(s)
    r  = @. clamp((s - pars.s_w) / max(pars.s_star - pars.s_w, eps(FT)), FT(0), FT(1))
    β  = @. pars.fww * r

    # gsw (leaf molar), all Fields
    ρw = FT(1000.0); Mw = FT(0.01801528)
    E_mps = @. β * E0
    E_mol = @. (ρw * E_mps) / Mw
    gsw_leaf = @. ifelse(VPD > eps(FT), min(E_mol * (P_air / VPD), pars.gsw_max), FT(0))

    # expose leaf/canopy molar conductance (optional aux)
    if Base.hasproperty(p.canopy.conductance, :gsw_leaf)
        @. p.canopy.conductance.gsw_leaf = gsw_leaf
    end
    if Base.hasproperty(p.canopy.conductance, :gsw_canopy)
        @. p.canopy.conductance.gsw_canopy = gsw_leaf * max(LAI, sqrt(eps(FT)))
    end

    # convert to m s^-1 and to canopy resistance Field
    g_leaf_mps = @. gsw_leaf * (Rgas * T_air / P_air)             # m s^-1 (leaf)
    g_canopy   = @. g_leaf_mps * max(LAI, sqrt(eps(FT)))           # m s^-1 (ground)
    @. p.canopy.conductance.r_stomata_canopy = 1 / (g_canopy + eps(FT))
    return nothing
end


# ===================== CWD-controlled OWUS =====================

# Maps w = [1, log1p(CWD_mm)] -> (α, a, b) via Γ (rows α,a,b), then
#   fww = σ(α), s_w = σ(a), s* = s_w + (1 - s_w) σ(b)
Base.@kwdef struct OWUSCWDStaticParameters{FT <: AbstractFloat}
    Γ::NTuple{6,FT}       # row-major [α0, αCWD, a0, aCWD, b0, bCWD]
    cwd_mm::FT            # site/pixel covariate (mm)
    gsw_max::FT = FT(Inf)
end
Base.eltype(::OWUSCWDStaticParameters{FT}) where {FT} = FT

struct OWUSConductanceCWDStatic{FT,
                                P<:OWUSCWDStaticParameters{FT}} <:
       AbstractStomatalConductanceModel{FT}
    parameters::P
end
OWUSConductanceCWDStatic{FT}(p::OWUSCWDStaticParameters{FT}) where {FT<:AbstractFloat} =
    OWUSConductanceCWDStatic{FT,typeof(p)}(p)


ClimaLand.auxiliary_vars(::OWUSConductanceCWDStatic) = (:r_stomata_canopy, :gsw_leaf, :gsw_canopy)
ClimaLand.auxiliary_types(::OWUSConductanceCWDStatic{FT}) where {FT} = (FT, FT, FT)
ClimaLand.auxiliary_domain_names(::OWUSConductanceCWDStatic) = (:surface, :surface, :surface)

@inline _σ(x) = inv(one(x) + exp(-x))
@inline _affine2(γ0::T, γ1::T, CWD::T) where {T} = γ0 + γ1 * log1p(CWD)

function update_canopy_conductance!(p, Y, model::OWUSConductanceCWDStatic, canopy)
    FT   = eltype(model.parameters)
    pars = model.parameters
    earth = canopy.parameters.earth_param_set
    Rgas  = LP.gas_constant(earth)

    # Drivers / state
    P_air = p.drivers.P
    T_air = p.drivers.T
    LAI   = p.canopy.biomass.area_index.leaf
    s     = _saturation_field(p, canopy, FT)
    E0    = _E0_field(p, canopy, FT)
    VPD   = _VPD_field(p, FT)

    # CWD → (α, a, b) → (fww, s*, sw)  (scalars of type FT)
    γα0, γαC, γa0, γaC, γb0, γbC = pars.Γ
    CWD = FT(pars.cwd_mm)
    α = γα0 + γαC * log1p(CWD)
    a = γa0  + γaC  * log1p(CWD)
    b = γb0  + γbC  * log1p(CWD)

    sw    = inv(FT(1) + exp(-a))
    sb    = inv(FT(1) + exp(-b))
    sstar = sw + (FT(1) - sw) * sb
    fww   = inv(FT(1) + exp(-α))

    # β(s) piecewise (broadcasted)
    r  = @. clamp((s - sw) / max(sstar - sw, eps(FT)), FT(0), FT(1))
    β  = @. fww * r

    # Convert to gsw (leaf, molar)
    ρw = FT(1000.0); Mw = FT(0.01801528)
    E_mps = @. β * E0
    E_mol = @. (ρw * E_mps) / Mw
    gsw_max = pars.gsw_max
    gsw_leaf = @. ifelse(VPD > eps(FT),
                         min(E_mol * (P_air / VPD), gsw_max),
                         FT(0))

    if Base.hasproperty(p.canopy.conductance, :gsw_leaf)
        @. p.canopy.conductance.gsw_leaf = gsw_leaf
    end
    if Base.hasproperty(p.canopy.conductance, :gsw_canopy)
        @. p.canopy.conductance.gsw_canopy = gsw_leaf * max(LAI, sqrt(eps(FT)))
    end

    g_leaf_mps = @. gsw_leaf * (Rgas * T_air / P_air)
    g_canopy   = @. g_leaf_mps * max(LAI, sqrt(eps(FT)))
    @. p.canopy.conductance.r_stomata_canopy = 1 / (g_canopy + eps(FT))
    return nothing
end
