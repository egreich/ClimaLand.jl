module SatisficingStomata

export SatisficingStomataModel, stomatal_conductance

# ──────────────────────────────────────────────────────────────────────────────
# Types & defaults
# ──────────────────────────────────────────────────────────────────────────────

"""
    SatisficingStomataModel(; safety_margin = 0.20,
                              min_photosynthesis_ratio = 0.70,
                              max_hydraulic_risk = 0.80,
                              frac_kcrit = 0.05,
                              selection = :minE)

A multi-criteria, “good-enough” stomatal module. It returns a single
operating point chosen from the satisficing set (by default the **lowest E**
that meets all constraints) and also reports the whole satisficing set.

Parameters
- `safety_margin`               :: minimum (Ecrit - E)/Ecrit
- `min_photosynthesis_ratio`    :: minimum A / max(A) along the E sweep
- `max_hydraulic_risk`          :: maximum tolerable `1 - K(P_leaf)/Kmax`
- `frac_kcrit`                  :: fraction of Kmax defining P_crit (for Ecrit)
- `selection`                   :: :minE (default), :maxA, or :median

Returned fields from `stomatal_conductance` include:
`E, gsw, A, Ci, Pleaf, safety_margin, hydraulic_risk, satisficing_set, ref_opt`.
"""
Base.@kwdef struct SatisficingStomataModel{T<:Real}
    safety_margin::T            = 0.20
    min_photosynthesis_ratio::T = 0.70
    max_hydraulic_risk::T       = 0.80
    frac_kcrit::T               = 0.05
    selection::Symbol           = :minE
end

# ──────────────────────────────────────────────────────────────────────────────
# Physiology helpers (Farquhar-lite) and hydraulics
# ──────────────────────────────────────────────────────────────────────────────

@inline A_farquhar(Ci, Vcmax, Jmax, Rd, Γ) = min(Vcmax * (Ci - Γ) / (Ci + 245.0),
                                                 Jmax  * (Ci - Γ) / (4.0*Ci + 8.0*Γ)) - Rd

"""
Solve for internal CO₂ (`Ci`) given transpiration rate `E` by matching supply and demand:
A(Ci) = E * (Ca - Ci) / (λ * D_unitless), with λ=1.6 (H₂O/CO₂ diffusivity ratio).

Returns `Ci`. Robust bisection on [Γ+ε, Ca-ε].
"""
function solve_Ci_given_E(E, Ca, D_unitless, Vcmax, Jmax, Rd; Γ=40.0, λ=1.6)
    if E <= 0
        return Ca
    end
    lo = Γ + 1e-6
    hi = Ca - 1e-6
    @inbounds for _ in 1:60
        mid = 0.5*(lo + hi)
        A_mid = A_farquhar(mid, Vcmax, Jmax, Rd, Γ)
        diff  = A_mid - E * (Ca - mid) / (λ * D_unitless)
        if diff > 0
            hi = mid
        else
            lo = mid
        end
    end
    return 0.5*(lo + hi)
end

@inline function A_from_E(E, Ca, D_unitless, Vcmax, Jmax, Rd; Γ=40.0)
    Ci = solve_Ci_given_E(E, Ca, D_unitless, Vcmax, Jmax, Rd; Γ=Γ)
    return A_farquhar(Ci, Vcmax, Jmax, Rd, Γ), Ci
end

function dA_dE_numeric(E, Ca, D_unitless, Vcmax, Jmax, Rd; rel_step=1e-4, Γ=40.0)
    dE = max(E * rel_step, 1e-8)
    A_plus, _  = A_from_E(E + dE, Ca, D_unitless, Vcmax, Jmax, Rd; Γ=Γ)
    A_minus, _ = A_from_E(E - dE, Ca, D_unitless, Vcmax, Jmax, Rd; Γ=Γ)
    return (A_plus - A_minus) / (2dE)
end

# Vulnerability curve & hydraulics
@inline vulnerability_curve(P, P50, a) = 1.0 / (1.0 + exp(-a * (P - P50)))             # logistic
@inline hydraulic_conductance(P, P50, a, Kmax) = Kmax * vulnerability_curve(P, P50, a)

function find_critical_P(P_soil, P50, a, Kmax; frac=0.05)
    # Scan down to -10 MPa; pick first P where K(P) <= frac*Kmax.
    P = range(P_soil, -10.0; length=2000)
    @inbounds for (i, p) in enumerate(P)
        if hydraulic_conductance(p, P50, a, Kmax) <= frac*Kmax
            return p
        end
    end
    return last(P)
end

"""
Solve K(P) * (P_soil - P) = E  for P via Newton iterations.
"""
function E_to_P(E, P_soil, P50, a, Kmax; tol=1e-8, maxiter=60)
    if E <= 0
        return P_soil
    end
    P = P_soil - E / max(Kmax, 1e-12)  # initial guess
    @inbounds for _ in 1:maxiter
        V  = vulnerability_curve(P, P50, a)
        K  = Kmax * V
        f  = K * (P_soil - P) - E
        if abs(f) < tol
            return P
        end
        sig   = exp(-a*(P - P50))
        dVdP  = (a*sig) / (1 + sig)^2
        dKdP  = Kmax * dVdP
        dfdP  = dKdP*(P_soil - P) - K
        if abs(dfdP) < 1e-14
            break
        end
        P -= f/dfdP
    end
    return P
end

@inline function Ecrit(P_soil, P50, a, Kmax; frac=0.05)
    Pcrit = find_critical_P(P_soil, P50, a, Kmax; frac=frac)
    max(0.0, Kmax * (P_soil - Pcrit))
end

# ──────────────────────────────────────────────────────────────────────────────
# Reference "Wang-optimal" point (optional diagnostic)
# ──────────────────────────────────────────────────────────────────────────────

"""
Minimize | dA/dE - A/(Ecrit - E) | (Wang-style marginal match) over E ∈ (0, 0.99 Ecrit).
Returns a NamedTuple with E, A, Ci, P_leaf, K_current, safety_margin.
"""
function wang_optimal(Ecrit, Ca, D_unitless, Vcmax, Jmax, Rd, P_soil, P50, a, Kmax)
    if Ecrit ≤ 0
        return nothing
    end
    function objective(E)
        (E ≤ 1e-8 || E ≥ 0.999Ecrit) && return Inf
        A, _ = A_from_E(E, Ca, D_unitless, Vcmax, Jmax, Rd)
        dA   = dA_dE_numeric(E, Ca, D_unitless, Vcmax, Jmax, Rd)
        dΘ   = A / (Ecrit - E)
        return abs(dA - dΘ)
    end
    # coarse grid search (robust & derivative-free)
    Egrid = range(0.005Ecrit, 0.95Ecrit; length=200)
    vals  = map(objective, Egrid)
    i     = argmin(vals)
    Eopt  = Float64(Egrid[i])

    Aopt, Ciopt = A_from_E(Eopt, Ca, D_unitless, Vcmax, Jmax, Rd)
    Pleaf = E_to_P(Eopt, P_soil, P50, a, Kmax)
    Kcur  = hydraulic_conductance(Pleaf, P50, a, Kmax)
    safety = (Ecrit - Eopt) / Ecrit
    return (E=Eopt, A=Aopt, Ci=Ciopt, P_leaf=Pleaf, K_current=Kcur,
            safety_margin=safety, type=:WangOptimal)
end

# ──────────────────────────────────────────────────────────────────────────────
# Satisficing scan and selection
# ──────────────────────────────────────────────────────────────────────────────

"""
Scan E ∈ (0.01, 0.99 Ecrit) and build metrics for each candidate point.
Returns a vector of NamedTuples: (:E, :A, :Ci, :P_leaf, :K_current, :safety_margin, :hydraulic_risk)
"""
function metrics_over_E(Ecrit, Ca, D_unitless, Vcmax, Jmax, Rd, P_soil, P50, a, Kmax; n=200)
    E_range = collect(range(0.01Ecrit, 0.99Ecrit; length=n))
    out = Vector{NamedTuple}(undef, n)
    maxA = -Inf
    for (j, E) in enumerate(E_range)
        A, Ci = A_from_E(E, Ca, D_unitless, Vcmax, Jmax, Rd)
        Pleaf = E_to_P(E, P_soil, P50, a, Kmax)
        Kcur  = hydraulic_conductance(Pleaf, P50, a, Kmax)
        safety = (Ecrit - E)/Ecrit
        risk = iszero(Kmax) ? 1.0 : (1.0 - Kcur/Kmax)
        nt = (E=E, A=A, Ci=Ci, P_leaf=Pleaf, K_current=Kcur,
              safety_margin=safety, hydraulic_risk=risk)
        out[j] = nt
        if A > maxA
            maxA = A
        end
    end
    return out, maxA
end

# ──────────────────────────────────────────────────────────────────────────────
# Pointwise SI satisficing (for broadcast-field callers)
# ──────────────────────────────────────────────────────────────────────────────

"""
    gsw_point_satisficing_SI(::Type{FT}, D_unitless, ca_mol, Vcmax, Jmax, Rd, Γstar_mol,
                              P_soil, P50, a_vuln, Kmax,
                              safety_margin, min_photosynthesis_ratio, max_hydraulic_risk, frac_kcrit,
                              selection_code; ngrid=128) -> gsw_leaf

Scalar (pointwise) satisficing solver in **SI units** returning leaf conductance
`gsw_leaf` in mol m⁻² s⁻¹. Designed to be called under broadcasting over Fields.
"""
@inline function gsw_point_satisficing_SI(
    ::Type{FT},
    D_unitless::FT,
    ca_mol::FT,
    Vcmax::FT,
    Jmax::FT,
    Rd::FT,
    Γstar_mol::FT,
    P_soil::FT,
    P50::FT,
    a_vuln::FT,
    Kmax::FT,
    safety_margin::FT,
    min_photosynthesis_ratio::FT,
    max_hydraulic_risk::FT,
    frac_kcrit::FT,
    selection_code::Int;
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
        return FT(0.5)*(lo + hi)
    end
    @inline A_from_E_local(E, Ca, Dstar, Vc, Jm, Rd) = begin
        Ci = solve_Ci(E, Ca, Dstar, Vc, Jm, Rd)
        return A_fq(Ci, Vc, Jm, Rd, Γstar_mol), Ci
    end

    # hydraulics (Pa)
    @inline vuln(P, P50, a) = inv(FT(1) + exp(-a * (P - P50)))
    @inline K_of_P(P)::FT    = Kmax * vuln(P, P50, a_vuln)

    # degenerate checks
    (Kmax ≤ FT(0)) && return FT(0)

    # Ecrit via simple scan to K=frac*Kmax
    steps = 512
    Pmin  = P_soil - FT(1e7)                     # down ~10 MPa
    dP    = (Pmin - P_soil) / (steps - 1)
    target = frac_kcrit * Kmax
    Pcrit = P_soil
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
                # rough median: pick near mid-count
                if satcnt == 1 + fld(ngrid, 4)
                    bestE = E; bestA = A
                end
            end
        end
    end

    if satcnt == 0
        bestE = FT(0.01) * Ecrit_local
    end

    # leaf gsw (mol m⁻² s⁻¹) via E ≈ gsw * D*
    return bestE / max(D_unitless, FT(1e-12))
end

# ──────────────────────────────────────────────────────────────────────────────
# High-level “reporting” solver
# ──────────────────────────────────────────────────────────────────────────────

"""
    stomatal_conductance(model::SatisficingStomataModel; kwargs...) -> NamedTuple

Compute a single operating point and report the satisficing set.

Required kwargs (typical canopy env/params):
- `Ca` (μmol mol⁻¹), `VPD` (kPa), `Patm` (kPa; default 101.325)
- `Vcmax`, `Jmax`, `Rd` (μmol m⁻² s⁻¹)
- `P_soil` (MPa), `P50` (MPa), `a` (unitless slope), `Kmax` (mol m⁻² s⁻¹ MPa⁻¹)
- optional: `Γ` (μmol mol⁻¹, default 40)

Returns: `(E, gsw, A, Ci, Pleaf, safety_margin, hydraulic_risk, satisficing_set, ref_opt, status)`.
"""
function stomatal_conductance(m::SatisficingStomataModel;
    Ca, VPD, Patm=101.325, Vcmax, Jmax, Rd,
    P_soil, P50, a, Kmax, Γ=40.0)

    D_unitless = VPD / Patm
    Ecrit_val = Ecrit(P_soil, P50, a, Kmax; frac=m.frac_kcrit)
    if Ecrit_val ≤ 1e-12
        # degenerate hydraulics: closed stomata
        return (E=0.0, gsw=0.0, A=0.0, Ci=Ca, Pleaf=P_soil, safety_margin=1.0,
                hydraulic_risk=1.0, satisficing_set=NamedTuple[], ref_opt=nothing,
                status=:closed)
    end

    ref = wang_optimal(Ecrit_val, Ca, D_unitless, Vcmax, Jmax, Rd, P_soil, P50, a, Kmax)

    mets, maxA = metrics_over_E(Ecrit_val, Ca, D_unitless, Vcmax, Jmax, Rd, P_soil, P50, a, Kmax)
    # satisficing filters
    sat = [x for x in mets if x.safety_margin ≥ m.safety_margin &&
                           x.A ≥ m.min_photosynthesis_ratio * maxA &&
                           x.hydraulic_risk ≤ m.max_hydraulic_risk]

    # gentle relaxation if empty
    if isempty(sat)
        sat = [x for x in mets if x.safety_margin ≥ 0.7*m.safety_margin &&
                               x.A ≥ 0.8*m.min_photosynthesis_ratio * maxA]
    end

    if isempty(sat)
        # fall back to conservative minimum E on entire curve
        choice = first(sort(mets, by = x->x.E))
        status = :fallback
    else
        choice = m.selection === :maxA  ? argmax_by(sat, x->x.A) :
                 m.selection === :median ? sat[clamp(round(Int, length(sat)/2), 1, length(sat))] :
                                           argmin_by(sat, x->x.E) # :minE default
        status = isempty([x for x in mets if x === choice]) ? :relaxed : :satisficing
    end

    E = choice.E
    gsw = E / max(D_unitless, 1e-12)      # E ≈ g_sw * D_unitless  → g_sw = E/D*
    return (E=E, gsw=gsw, A=choice.A, Ci=choice.Ci, Pleaf=choice.P_leaf,
            safety_margin=choice.safety_margin, hydraulic_risk=choice.hydraulic_risk,
            satisficing_set=sat, ref_opt=ref, status=status)
end

# tiny utils
@inline argmin_by(v, f) = v[findmin((f(x) for x in v)...)[2]]
@inline argmax_by(v, f) = v[findmax((f(x) for x in v)...)[2]]

end # module
