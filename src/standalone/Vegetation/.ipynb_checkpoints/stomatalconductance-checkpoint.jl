export MedlynConductanceParameters,
    MedlynConductanceModel, PModelConductanceParameters, PModelConductance, SatisficingConductanceParameters, SatisficingConductance

abstract type AbstractStomatalConductanceModel{FT} <:
              AbstractCanopyComponent{FT} end

"""
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
    area_index = p.canopy.hydraulics.area_index
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
    function MedlynConductanceParameters(::Type{FT};
        g1 = 790,
        kwargs...
    )
    function MedlynConductanceParameters(toml_dict;
        g1 = 790,
        kwargs...
    )

Floating-point and toml dict based constructor supplying default values
for the MedlynConductanceParameters struct.
Additional parameter values can be directly set via kwargs.
"""
MedlynConductanceParameters(::Type{FT}; kwargs...) where {FT <: AbstractFloat} =
    MedlynConductanceParameters(CP.create_toml_dict(FT); kwargs...)

function MedlynConductanceParameters(
    toml_dict::CP.AbstractTOMLDict;
    g1 = 790,
    kwargs...,
)
    name_map = (;
        :relative_diffusivity_of_water_vapor => :Drel,
        :min_stomatal_conductance => :g0,
    )

    parameters = CP.get_parameter_values(toml_dict, name_map, "Land")
    FT = CP.float_type(toml_dict)
    g1 = FT.(g1)
    G1 = typeof(g1)
    return MedlynConductanceParameters{FT, G1}(; g1, parameters..., kwargs...)
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
    area_index = p.canopy.hydraulics.area_index
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

#################### Satisficing model conductance ####################

"""
    SatisficingConductanceParameters{FT<:AbstractFloat}

Required parameters for the satisficing stomatal conductance model.
$(DocStringExtensions.FIELDS)
"""
# Map user-friendly symbols to internal codes
_selection_code(x::Symbol) = x === :minE   ? 1 :
                             x === :maxA   ? 2 :
                             x === :median ? 3 :
                             throw(ArgumentError("Unknown selection = $x. Use :minE, :maxA, or :median."))
_selection_code(x::Integer) = Int(x)

Base.@kwdef struct SatisficingConductanceParameters{FT}
    "Relative diffusivity of water vapor (unitless); used in E–gₛ conversion"
    Drel::FT
    "Minimum stomatal conductance (mol m⁻² s⁻¹); kept for API compatibility"
    g0::FT = zero(FT)
    "Slope parameter (Pa¹ᐟ²); kept for API compatibility with Medlyn-based code"
    g1::FT = zero(FT)
    "Minimum safety margin: (Ecrit - E)/Ecrit must be ≥ this"
    safety_margin::FT            = FT(0.20)
    "Minimum photosynthesis ratio: A must be ≥ this × max(A)"
    min_photosynthesis_ratio::FT = FT(0.70)
    "Maximum tolerated hydraulic risk = 1 - K(Pleaf)/Kmax"
    max_hydraulic_risk::FT       = FT(0.80)
    "Fraction of Kmax that defines hydraulic critical point Ecrit"
    frac_kcrit::FT               = FT(0.05)
    "Selection rule for a single operating point: 1=minE, 2=maxA, 3=median"
    selection_code::Int          = 1
end

Base.eltype(::SatisficingConductanceParameters{FT}) where {FT} = FT

# Keyword constructor that accepts `selection` (Symbol/Int) OR `selection_code`
function SatisficingConductanceParameters{FT}(;
    Drel::FT,
    g0::FT                       = zero(FT), # API compatibility
    g1::FT                       = zero(FT), # API compatibility
    safety_margin::FT            = FT(0.20),
    min_photosynthesis_ratio::FT = FT(0.70),
    max_hydraulic_risk::FT       = FT(0.80),
    frac_kcrit::FT               = FT(0.05),
    selection::Union{Symbol,Integer} = :minE,
    selection_code::Union{Nothing,Integer} = nothing,
) where {FT<:AbstractFloat}
    code = selection_code === nothing ? _selection_code(selection) : Int(selection_code)
    return SatisficingConductanceParameters{FT}(
        Drel, g0, g1,
        safety_margin,
        min_photosynthesis_ratio,
        max_hydraulic_risk,
        frac_kcrit,
        code,
    )
end

struct SatisficingConductance{FT, SCP <: SatisficingConductanceParameters{FT}} <:
       AbstractStomatalConductanceModel{FT}
    parameters::SCP
end

ClimaLand.auxiliary_vars(::SatisficingConductance) = (:r_stomata_canopy,)
ClimaLand.auxiliary_types(::SatisficingConductance{FT}) where {FT} = (FT,)
ClimaLand.auxiliary_domain_names(::SatisficingConductance) = (:surface,)

# ──────────────────────────────────────────────────────────────────────────────
# Broadcast-safe helpers (operate under @. and accept Fields or scalars)
# ──────────────────────────────────────────────────────────────────────────────

@inline kPa(P) = P > 200 ? P * 1e-3 : P  # Pa→kPa if needed; leaves kPa unchanged

# VPD (kPa) from temperature (K), specific humidity (kg/kg), pressure (Pa or kPa)
@inline function vpd_kPa(T, q, _thermo, P)
    Tc = T - 273.15
    es = 0.611 * exp(17.27 * Tc / (Tc + 237.3))           # kPa (Tetens)
    Pk = kPa(P)                                           # kPa
    ε  = 0.622
    e  = (q * Pk) / (ε + (1 - ε) * q)                     # kPa
    max(es - e, 0.0)
end

# Pointwise satisficing solver in SI units (leaf-level), used under broadcasting.
# Inputs must already be in SI and leaf-level:
#  D_unitless (VPD/Patm), ca_mol (mol/mol), Vcmax/Jmax/Rd (mol m⁻² s⁻¹),
#  Γstar (mol/mol), P_soil/P50 (Pa), a_vuln (1/Pa), Kmax (mol m⁻² s⁻¹ Pa⁻¹),
#  and model thresholds; returns gsw_leaf (mol m⁻² s⁻¹).
@inline function gsw_point_satisficing_SI(
    ::Type{FT},
    D_unitless, ca_mol, Vcmax, Jmax, Rd, Γstar_mol,
    P_soil, P50, a_vuln, Kmax,
    safety_margin, min_photosynthesis_ratio, max_hydraulic_risk, frac_kcrit,
    selection_code;
    ngrid::Int = 128,
) where {FT<:Real}

    # local Farquhar-lite (SI)
    @inline A_fq(Ci, Vc, Jm, Rd, Γ) = min(Vc * (Ci - Γ) / (Ci + FT(245.0)),
                                          Jm * (Ci - Γ) / (FT(4)*Ci + FT(8)*Γ)) - Rd
    @inline function solve_Ci(E, Ca, Dstar, Vc, Jm, Rd; Γ=Γstar_mol, λ=FT(1.6))
        if E ≤ 0
            return Ca
        end
        lo = Γ + FT(1e-12)
        hi = max(Ca - FT(1e-12), lo + FT(1e-10))
        @inbounds for _ in 1:60
            mid = FT(0.5)*(lo + hi)
            A_mid = A_fq(mid, Vc, Jm, Rd, Γ)
            diff  = A_mid - E * (Ca - mid) / (λ * max(Dstar, FT(1e-12)))
            if diff > 0
                hi = mid
            else
                lo = mid
            end
        end
        FT(0.5)*(lo + hi)
    end
    @inline A_from_E_local(E, Ca, Dstar, Vc, Jm, Rd) = begin
        Ci = solve_Ci(E, Ca, Dstar, Vc, Jm, Rd)
        A_fq(Ci, Vc, Jm, Rd, Γstar_mol), Ci
    end

    # hydraulics (Pa)
    @inline vuln(P, P50, a) = inv(FT(1) + exp(-a * (P - P50)))
    @inline K_of_P(P) = Kmax * vuln(P, P50, a_vuln)

    # degenerate
    (Kmax ≤ FT(0)) && return FT(0)

    # Ecrit via scan to K = frac*Kmax
    steps  = 256
    Pmin   = P_soil - FT(1e7)                       # ~10 MPa below soil
    dP     = (Pmin - P_soil) / (steps - 1)
    target = frac_kcrit * Kmax
    Pcrit  = P_soil
    @inbounds for i in 1:steps
        P = P_soil + dP*(i-1)
        if K_of_P(P) ≤ target
            Pcrit = P
            break
        end
    end
    Ecrit_local = max(FT(0), target * (P_soil - Pcrit))
    (Ecrit_local ≤ FT(1e-20)) && return FT(0)

    # first pass: maxA
    maxA = -typemax(FT)
    @inbounds for j in 1:ngrid
        E = (FT(0.01) + (FT(0.98)*FT(j-1)/(ngrid-1))) * Ecrit_local
        A, _ = A_from_E_local(E, ca_mol, D_unitless, Vcmax, Jmax, Rd)
        maxA = A > maxA ? A : maxA
    end
    maxA = max(maxA, FT(0))

    want_minE   = selection_code == 1
    want_maxA   = selection_code == 2
    want_median = selection_code == 3

    bestE  = FT(0)
    bestA  = FT(0)
    satcnt = 0

    # second pass: filter & select
    @inbounds for j in 1:ngrid
        E = (FT(0.01) + (FT(0.98)*FT(j-1)/(ngrid-1))) * Ecrit_local
        A, _ = A_from_E_local(E, ca_mol, D_unitless, Vcmax, Jmax, Rd)
        # quick fixed-point update for P at this E
        P = P_soil
        @inbounds for _ in 1:20
            K = K_of_P(P)
            K ≤ FT(1e-20) && break
            Pnew = P_soil - E / K
            P = (FT(0.5))*P + (FT(0.5))*Pnew
        end
        Kcur   = K_of_P(P)
        safety = (Ecrit_local - E) / Ecrit_local
        risk   = FT(1) - Kcur / max(Kmax, FT(1e-20))

        ok = (safety ≥ safety_margin) &
             (A ≥ min_photosynthesis_ratio * maxA) &
             (risk ≤ max_hydraulic_risk)

        if ok
            satcnt += 1
            if want_minE
                bestE = E; bestA = A
                break
            elseif want_maxA
                if A > bestA
                    bestA = A; bestE = E
                end
            else
                # rough median
                if satcnt == 1 + fld(ngrid, 4)
                    bestE = E; bestA = A
                end
            end
        end
    end

    if satcnt == 0
        bestE = FT(0.01) * Ecrit_local
    end

    # E ≈ g_sw * D*  → g_sw = E / D*
    bestE / max(D_unitless, FT(1e-12))
end

# --- scalar Farquhar-lite pieces (FD-free) ---
@inline A_farquhar_scalar(Ci, Vcmax, Jmax, Rd, Γ) =
    min(Vcmax * (Ci - Γ) / (Ci + 245.0),
        Jmax  * (Ci - Γ) / (4.0*Ci + 8.0*Γ)) - Rd

@inline function solve_Ci_given_E_scalar(E, Ca, D_unitless, Vcmax, Jmax, Rd; Γ=40.0, λ=1.6)
    if E <= 0
        return Ca
    end
    lo = Γ + 1e-6
    hi = Ca - 1e-6
    @inbounds for _ in 1:60
        mid  = 0.5*(lo + hi)
        Amd  = A_farquhar_scalar(mid, Vcmax, Jmax, Rd, Γ)
        diff = Amd - E*(Ca - mid)/(λ*D_unitless)
        if diff > 0
            hi = mid
        else
            lo = mid
        end
    end
    return 0.5*(lo + hi)
end

@inline function A_from_E_scalar(E, Ca, D_unitless, Vcmax, Jmax, Rd; Γ=40.0)
    Ci = solve_Ci_given_E_scalar(E, Ca, D_unitless, Vcmax, Jmax, Rd; Γ=Γ)
    return A_farquhar_scalar(Ci, Vcmax, Jmax, Rd, Γ), Ci
end

# Hydraulics (logistic vulnerability), all scalar:
@inline vulnerability_curve_scalar(P, P50, a) = 1.0 / (1.0 + exp(-a*(P - P50)))
@inline hydraulic_conductance_scalar(P, P50, a, Kmax) = Kmax * vulnerability_curve_scalar(P, P50, a)

@inline function find_critical_P_scalar(P_soil, P50, a, Kmax; frac=0.05)
    # Scan with fixed length so it’s deterministic and broadcast-safe when called from scalar kernel
    P = range(P_soil, P_soil - 10.0; length=2000)
    @inbounds for p in P
        if hydraulic_conductance_scalar(p, P50, a, Kmax) <= frac*Kmax
            return p
        end
    end
    return last(P)
end

@inline function Ecrit_scalar(P_soil, P50, a, Kmax; frac=0.05)
    Pcrit = find_critical_P_scalar(P_soil, P50, a, Kmax; frac=frac)
    max(0.0, Kmax * (P_soil - Pcrit))
end

@inline function E_to_P_scalar(E, P_soil, P50, a, Kmax; tol=1e-8, maxiter=50)
    if E <= 0
        return P_soil
    end
    P = P_soil - E / max(Kmax, 1e-12)
    @inbounds for _ in 1:maxiter
        V  = vulnerability_curve_scalar(P, P50, a)
        K  = Kmax * V
        f  = K*(P_soil - P) - E
        if abs(f) < tol
            return P
        end
        sig  = exp(-a*(P - P50))
        dVdP = (a*sig) / (1 + sig)^2
        dKdP = Kmax*dVdP
        dfdP = dKdP*(P_soil - P) - K
        if abs(dfdP) < 1e-14
            break
        end
        P -= f/dfdP
    end
    return P
end

@inline function _gsw_from_env_scalar(
    Ca_umol, D_unitless, Vcmax, Jmax, Rd, Γstar_umol,
    P_soil, P50, a_vuln, Kmax,
    safety_margin, min_A_ratio, max_hydraulic_risk, frac_kcrit,
    selection_code::Int,
)
    # Units: Ca, Γstar in μmol/mol → convert to mol/mol
    Ca = Ca_umol * 1e-6
    Γ  = Γstar_umol * 1e-6

    # If D≈0 or Kmax≈0, close stomata
    if D_unitless ≤ 1e-12 || Kmax ≤ 1e-20
        return 0.0
    end

    Ecrit = Ecrit_scalar(P_soil, P50, a_vuln, Kmax; frac=frac_kcrit)
    if Ecrit ≤ 1e-16
        return 0.0
    end

    # Scan a fixed grid in (0.01,0.99) * Ecrit and pick an operating point by rule
    n   = 200
    Emin, Emax = 0.01*Ecrit, 0.99*Ecrit
    dE  = (Emax - Emin) / (n - 1)

    maxA = -Inf
    # First pass: get max A for ratio
    @inbounds for j in 0:n-1
        E = Emin + j*dE
        A, _ = A_from_E_scalar(E, Ca, D_unitless, Vcmax, Jmax, Rd; Γ=Γ)
        if A > maxA
            maxA = A
        end
    end
    if !isfinite(maxA) || maxA ≤ 0
        return 0.0
    end

    # Second pass: apply satisficing filters and choose candidate
    chosen_E  = NaN
    chosen_A  = -Inf
    # For :median we store counts, but since we need fixed memory, approximate with halfway-kth pick
    count_sat = 0
    median_target = (n + 1) ÷ 2
    median_E = Emin

    @inbounds for j in 0:n-1
        E = Emin + j*dE
        A, _     = A_from_E_scalar(E, Ca, D_unitless, Vcmax, Jmax, Rd; Γ=Γ)
        Pleaf    = E_to_P_scalar(E, P_soil, P50, a_vuln, Kmax)
        Kcur     = hydraulic_conductance_scalar(Pleaf, P50, a_vuln, Kmax)
        safety   = (Ecrit - E)/Ecrit
        risk     = (Kmax ≤ 0) ? 1.0 : (1.0 - Kcur/max(Kmax,1e-30))

        ok = (safety ≥ safety_margin) & (A ≥ min_A_ratio*maxA) & (risk ≤ max_hydraulic_risk)
        if ok
            count_sat += 1
            if selection_code == 1  # :minE
                chosen_E = E
                chosen_A = A
                break
            elseif selection_code == 2  # :maxA
                if A > chosen_A
                    chosen_A = A
                    chosen_E = E
                end
            else # :median (approximate by the mid-ranked seen so far)
                if count_sat == median_target
                    median_E = E
                end
            end
        end
    end

    if selection_code == 3  # :median
        if count_sat > 0
            chosen_E = median_E
        end
    end

    # If nothing satisfied, fallback to most conservative point on the curve
    if !isfinite(chosen_E)
        chosen_E = Emin
    end

    # g_sw ≈ E / D_unitless  (mol m⁻² s⁻¹)
    gsw = chosen_E / D_unitless
    return max(gsw, 0.0)
end

# Broadcast-friendly wrapper: same signature, just calls the scalar kernel
@inline _gsw_from_env(args...) = _gsw_from_env_scalar(args...)

"""
    update_canopy_conductance!(p, Y, model::SatisficingConductance, canopy)

Computes and updates canopy-level stomatal conductance (m s⁻¹) via the
multi-criteria “satisficing” approach
"""
function update_canopy_conductance!(p, Y, model::SatisficingConductance, canopy)

    FT   = eltype(model.parameters)
    earth = canopy.parameters.earth_param_set
    R     = LP.gas_constant(earth)
    thermo = earth.thermo_params

    # Drivers (Fields)
    c_co2_air = p.drivers.c_co2   # mol/mol
    P_air     = p.drivers.P       # Pa (or kPa)
    T_air     = p.drivers.T       # K
    q_air     = p.drivers.q       # kg/kg

    # LAI (Field)
    LAI = p.canopy.hydraulics.area_index.leaf


    # Moisture stress β from leaf compartment
    fp = canopy.photosynthesis.parameters
    ψ      = p.canopy.hydraulics.ψ  # Field of Tuple
    n_stem = canopy.hydraulics.n_stem
    n_leaf = canopy.hydraulics.n_leaf
    i_end  = n_stem + n_leaf
    ρ_l    = LP.ρ_cloud_liq(earth)
    grav   = LP.grav(earth)
    sc, pc = fp.sc, fp.pc
    
    ψ_leaf = ψ.:($i_end)   # last compartment (leaf)
    ψ_leaf_Pa = @. lazy(ψ_leaf * ρ_l * grav)
    β = @. lazy(moisture_stress(ψ_leaf_Pa, sc, pc))

    # Canopy temperature (Field)
    T_canopy = canopy_temperature(canopy.energy, canopy, Y, p)

    # Hydraulics: soil pressure (use first compartment), convert to Pa
    P_soil = @. lazy(ψ.:(1) * ρ_l * grav)  # Pa

    # Photosynthesis parameters (scalars or Fields)
    Γstar = @. lazy(co2_compensation_farquhar(fp.Γstar25, fp.ΔHΓstar, T_canopy, fp.To, R))
    Jmax  = @. lazy(max_electron_transport_farquhar(fp.Vcmax25, fp.ΔHJmax, T_canopy, fp.To, R))
    Vcmax = @. lazy(compute_Vcmax_farquhar(fp.is_c3, fp.Vcmax25, T_canopy, R, fp.To, fp.ΔHVcmax, fp.Q10, fp.s1, fp.s2, fp.s3, fp.s4))
    β     = @. lazy(moisture_stress(ψ_leaf_Pa, fp.sc, fp.pc))
    Rd    = @. lazy(dark_respiration_farquhar(fp.is_c3, fp.Vcmax25, β, T_canopy, R, fp.To, fp.fC3, fp.ΔHRd, fp.Q10, fp.s5, fp.s6, fp.fC4))

    # Hydraulics parameters from conductivity model (scalars; broadcast naturally)
    cm   = canopy.hydraulics.parameters.conductivity_model
    Kmax = hasproperty(cm, :K_sat) ? getproperty(cm, :K_sat) :
           hasproperty(cm, :Kmax)  ? getproperty(cm, :Kmax)  :
           error("Conductivity model has no K_sat/Kmax; fields are $(propertynames(cm))")
    a_vuln = hasproperty(cm, :c) ? getproperty(cm, :c) :
             hasproperty(cm, :a) ? getproperty(cm, :a) :
             error("Conductivity model has no c/a; fields are $(propertynames(cm))")
    P50 = hasproperty(cm, :ψ50) ? getproperty(cm, :ψ50) :
          hasproperty(cm, :P50) ? getproperty(cm, :P50) :
          hasproperty(cm, :ψ63) ? getproperty(cm, :ψ63) :
          error("No ψ50/P50/ψ63 on conductivity model; fields are $(propertynames(cm))")

    # VPD*, D (unitless)
    VPDk = @. lazy(vpd_kPa(T_air, q_air, thermo, P_air))
    Patm = @. lazy(kPa(P_air))
    Dstar = @. lazy(VPDk / max(Patm, sqrt(eps(FT))))   # VPD/Patm

    # Unpack satisficing parameters
    (; safety_margin, min_photosynthesis_ratio, max_hydraulic_risk, frac_kcrit, selection_code) = model.parameters

    # Broadcast satisficing kernel to get gsw (Field, mol m⁻² s⁻¹)
    #    Note: Ca, Γstar expected as μmol/mol → multiply by 1e6
    gsw = @. lazy(_gsw_from_env(
        c_co2_air * 1e6, Dstar, Vcmax, Jmax, Rd, Γstar * 1e6,
        P_soil, P50, a_vuln, Kmax,
        model.parameters.safety_margin,
        model.parameters.min_photosynthesis_ratio,
        model.parameters.max_hydraulic_risk,
        model.parameters.frac_kcrit,
        model.parameters.selection_code,
    ))
    
    # 4) Convert gsw → m s⁻¹ and write r_stomata_canopy
    gsw_canopy_mps = @. lazy(conductance_molar_flux_to_m_per_s(FT(gsw), T_air, R, P_air) * max(LAI, sqrt(eps(FT))))
    @. p.canopy.conductance.r_stomata_canopy = 1 / (gsw_canopy_mps + eps(FT))
end

