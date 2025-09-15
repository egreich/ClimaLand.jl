#!/usr/bin/env julia
using Test
import ClimaLand
using ClimaLand: PrescribedAtmosphere, PrescribedRadiativeFluxes, PrescribedGroundConditions
using ClimaLand.Canopy
using ClimaLand.Canopy.PlantHydraulics
using ClimaLand.Domains: Point
import ClimaLand.Parameters as LP
using ClimaUtilities.TimeVaryingInputs: TimeVaryingInput
using Dates

# ---- Prescribed forcing (no data regridding) ----
const START = DateTime(2005)
shortwave_radiation(t; kwargs...) = 800.0
longwave_radiation(t) = 300.0
u_atmos(t) = 2.0
liquid_precip(t) = 0.0
snow_precip(t) = 0.0
T_atmos(t) = 293.0
q_atmos(t) = 0.010
P_atmos(t) = 1e5
c_atmos(t) = 4.20e-4
zenith_angle = (t, s) -> 0.5


@testset "Satisficing conductance smoke test (no CLM regridding)" begin
  for FT in (Float32, Float64)
    # Domain & params
    domain = Point(; z_sfc = FT(0))
    earth = LP.LandParameters(FT)
    shared = Canopy.SharedCanopyParameters{FT, typeof(earth)}(FT(0.2), FT(0.02), earth)

    # Atmos & radiation (prescribed)
    atmos = PrescribedAtmosphere(
      TimeVaryingInput(liquid_precip),
      TimeVaryingInput(snow_precip),
      TimeVaryingInput(T_atmos),
      TimeVaryingInput(u_atmos),
      TimeVaryingInput(q_atmos),
      TimeVaryingInput(P_atmos),
      START, FT(2), earth; c_co2 = TimeVaryingInput(c_atmos),
    )
    radiation = PrescribedRadiativeFluxes(
      FT, TimeVaryingInput(shortwave_radiation), TimeVaryingInput(longwave_radiation),
      START; θs = zenith_angle, earth_param_set = earth,
    )
    ground = PrescribedGroundConditions{FT}()

    # Photosynthesis: explicit scalar parameters (avoids CLM regridding)
    photosyn_params = Canopy.FarquharParameters(FT, FT(1);  # is_c3 = 1
      Vcmax25 = FT(9e-5),                                   # mol m^-2 s^-1
      sc = LP.get_default_parameter(FT, :low_water_pressure_sensitivity),
      pc = LP.get_default_parameter(FT, :moisture_stress_ref_water_pressure),
    )
    photosynthesis = Canopy.FarquharModel{FT}(photosyn_params)

    # Radiation model: explicit scalar parameters (avoids CLM regridding)
    bl_params = Canopy.BeerLambertParameters(FT;
      α_PAR_leaf = FT(0.10), α_NIR_leaf = FT(0.40),
      ϵ_canopy = LP.get_default_parameter(FT, :canopy_emissivity),
    )
    rt_model = Canopy.BeerLambertModel{FT}(bl_params)

    # Hydraulics: ALL explicit scalars (no TOML)
    LAI = TimeVaryingInput(t -> FT(3))
    ν   = FT(0.7)                   # m3/m3
    S_s = FT(1e-2 * 0.0098)         # m3/m3 per m
    K_sat_plant = FT(1.8e-8)        # m/s
    ψ63 = FT(-4 / 0.0098)           # MPa→m conversion applied in model
    Weibull_param = FT(4)           # unitless
    a_bulk = FT(0.05 * 0.0098)      # m
    rooting_depth = FT(0.5)         # m

    conductivity_model = PlantHydraulics.Weibull{FT}(K_sat_plant, ψ63, Weibull_param)
    retention_model    = PlantHydraulics.LinearRetentionCurve{FT}(a_bulk)

    ph_params = PlantHydraulics.PlantHydraulicsParameters(;
      ai_parameterization = PlantHydraulics.PrescribedSiteAreaIndex{FT}(LAI, FT(0), FT(1)),
      ν = ν, S_s = S_s,
      rooting_depth = rooting_depth,
      conductivity_model = conductivity_model,
      retention_model = retention_model,
    )

    hydraulics = Canopy.PlantHydraulicsModel{FT}(;
      n_stem = 0, n_leaf = 1,
      compartment_midpoints = [FT(0.5)], compartment_surfaces = [FT(0.0), FT(1.0)],
      parameters = ph_params,
      transpiration = PlantHydraulics.DiagnosticTranspiration{FT}(),
    )

    # Energy & respiration
    energy = Canopy.BigLeafEnergyModel{FT}(Canopy.BigLeafEnergyParameters{FT}(FT(2e3)))
    ar     = Canopy.AutotrophicRespirationModel{FT}(Canopy.AutotrophicRespirationParameters(FT))
    sif    = Canopy.Lee2015SIFModel{FT}()

    # Satisficing conductance
    conductance = Canopy.SatisficingConductance{FT}(;
        Drel = FT(1.6),
        safety_margin = FT(0.25),
        min_photosynthesis_ratio = FT(0.70),
        max_hydraulic_risk = FT(0.80),
        frac_kcrit = FT(0.05),
        selection = :minE,   #1
    )


    # Build canopy
    bc = Canopy.AtmosDrivenCanopyBC(atmos, radiation, ground)
    canopy = Canopy.CanopyModel{FT}(;
      autotrophic_respiration = ar,
      radiative_transfer = rt_model,
      photosynthesis = photosynthesis,
      conductance = conductance,
      hydraulics = hydraulics,
      energy = energy,
      sif = sif,
      boundary_conditions = bc,
      parameters = shared,
      domain = domain,
    )

    # Initialize, update once
    Y, p, coords = ClimaLand.initialize(canopy)
    set_initial_cache! = ClimaLand.make_set_initial_cache(canopy)
    update_aux!       = ClimaLand.make_update_aux(canopy)
    compute_exp!      = ClimaLand.make_compute_exp_tendency(canopy)

    t0 = FT(0)
    set_initial_cache!(p, Y, t0)
    dY = similar(Y)
    compute_exp!(dY, Y, p, t0)
    update_aux!(p, Y, t0)

    # Assertions
    r_can = Array(parent(p.canopy.conductance.r_stomata_canopy))[1]  # s m^-1
    @test isfinite(r_can) && r_can ≥ FT(0)
    g_can = 1 / max(r_can, eps(FT))                                  # m s^-1
    @test isfinite(g_can) && g_can ≥ FT(0)

    An = Array(parent(p.canopy.photosynthesis.An))[1]
    @test isfinite(An)

    ET = Array(parent(p.canopy.turbulent_fluxes.transpiration))[1]
    @test isfinite(ET) && ET ≥ FT(0)

    println("[$(FT)] g_can = $(g_can) m s^-1, An = $(An) mol m^-2 s^-1, ET = $(ET) kg m^-2 s^-1")
  end
end
