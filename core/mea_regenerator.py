"""
=============================================================================
MODULE 3: MEA-Based CO₂ Regenerator (CCUS Thermal Model)
=============================================================================
Project : GX×DX Closed-Loop Digital Twin
Author  : V Niranjana | IIT Jodhpur
Purpose : Models the thermal energy requirements of a monoethanolamine (MEA)
          CO₂ capture and regeneration unit.

          The regenerator strips CO₂ from CO₂-rich MEA solvent by applying
          heat. This module calculates:
            1. Total heat required to regenerate MEA (sensible + reaction)
            2. How much of that heat is supplied by waste heat (glycol loop)
            3. How much supplemental solar thermal heat is still needed
            4. CO₂ capture rate and tonnes captured per hour
            5. Carbon credit value generated

Two-Stage Heating Model:
  Stage 1 — Waste Heat (glycol loop, ~25–36°C):
            Pre-heats MEA solution from T_ambient → T_glycol_out
  Stage 2 — Solar Thermal supplement (~36°C → 120°C):
            Completes regeneration to stripping temperature

Key Chemistry:
  MEA + CO₂ + H₂O ⇌ MEA-carbamate  (absorption, exothermic)
  Heat reverses this reaction        (regeneration, endothermic)
  ΔH_absorption ≈ 85 kJ/mol CO₂    (energy to break MEA-CO₂ bond)

Reference flue gas: Industrial boiler exhaust
  ~12% CO₂ by volume, 1000 Nm³/h flow rate
=============================================================================
"""

import numpy as np

# ---------------------------------------------------------------------------
# PHYSICAL & CHEMICAL CONSTANTS
# ---------------------------------------------------------------------------

# MEA solution properties (30 wt% aqueous MEA — industry standard)
MEA_CONCENTRATION   = 0.30          # 30 wt% MEA in water
MEA_CP              = 3.80          # kJ/kg·K  specific heat of 30% MEA solution
MEA_DENSITY         = 1020.0        # kg/m³    density of 30% MEA solution

# Thermodynamics of CO₂-MEA reaction
DELTA_H_ABS         = 85.0          # kJ/mol CO₂  heat of absorption/desorption
CO2_MOLAR_MASS      = 44.01         # g/mol
MEA_MOLAR_MASS      = 61.08         # g/mol

# Process temperatures
T_AMBIENT           = 25.0          # °C  — MEA enters absorber at ambient
T_REGEN             = 120.0         # °C  — standard MEA regeneration temperature
                                    #       (reboiler operates at 120–130°C)
T_LEAN_RETURN       = 40.0          # °C  — lean MEA returned to absorber after cooling

# MEA carrying capacity
MEA_LOADING_RICH    = 0.45          # mol CO₂/mol MEA — rich solvent (absorber exit)
MEA_LOADING_LEAN    = 0.20          # mol CO₂/mol MEA — lean solvent (regen exit)
DELTA_LOADING       = MEA_LOADING_RICH - MEA_LOADING_LEAN  # = 0.25 mol CO₂/mol MEA

# Carbon credit market
CARBON_CREDIT_PRICE = 85.0          # USD per tonne CO₂ (EU ETS ~85 USD, 2024)

# Flue gas parameters (reference industrial boiler)
FLUE_GAS_FLOW_NM3H  = 1000.0        # Nm³/h  — normal cubic meters per hour
CO2_FRACTION_FLUE   = 0.12          # 12 vol% CO₂ in flue gas (typical boiler)
CO2_DENSITY_NM3     = 1.977         # kg/Nm³ at 0°C, 1 atm

# ---------------------------------------------------------------------------
# CORE CLASS
# ---------------------------------------------------------------------------

class MEARegenerator:
    """
    Models a MEA-based CO₂ capture and regeneration unit.

    Parameters
    ----------
    mea_flow_kgs : float
        Mass flow rate of MEA solution through regenerator [kg/s].
        Default = 2.0 kg/s (sized for 100kW data center flue gas equivalent).
    target_capture_rate : float
        Desired CO₂ capture efficiency (fraction). Default = 0.90 (90%).
    flue_gas_flow_nm3h : float
        Flue gas volumetric flow rate [Nm³/h]. Default = 1000 Nm³/h.
    co2_fraction : float
        CO₂ volume fraction in flue gas. Default = 0.12 (12%).
    carbon_price_usd : float
        Carbon credit price [USD/tonne CO₂]. Default = 85 USD/t.
    """

    def __init__(
        self,
        mea_flow_kgs        = 0.21,
        target_capture_rate = 0.90,
        flue_gas_flow_nm3h  = FLUE_GAS_FLOW_NM3H,
        co2_fraction        = CO2_FRACTION_FLUE,
        carbon_price_usd    = CARBON_CREDIT_PRICE
    ):
        self.mea_flow_kgs        = mea_flow_kgs
        self.target_capture_rate = target_capture_rate
        self.flue_gas_flow_nm3h  = flue_gas_flow_nm3h
        self.co2_fraction        = co2_fraction
        self.carbon_price_usd    = carbon_price_usd

        # Pre-compute CO₂ available in flue gas [kg/h]
        co2_flow_nm3h       = flue_gas_flow_nm3h * co2_fraction
        self.co2_available_kgh = co2_flow_nm3h * CO2_DENSITY_NM3   # kg/h

        # Max capturable CO₂ at target rate [kg/h]
        self.co2_target_kgh = self.co2_available_kgh * target_capture_rate

    # ------------------------------------------------------------------
    def compute_heat_requirement(self, T_inlet_c: float = T_AMBIENT) -> dict:
        """
        Compute total heat required to fully regenerate MEA from T_inlet_c
        to regeneration temperature (120°C).

        Parameters
        ----------
        T_inlet_c : float
            Temperature of MEA solution entering the regenerator [°C].
            This is the glycol outlet temperature from the HX module.

        Returns
        -------
        dict with full heat requirement breakdown.
        """

        # --- Step 1: Sensible heat — heating MEA solution to regen temp ---
        # Q_sensible = ṁ × Cp × ΔT
        # ṁ in kg/s, Cp in kJ/kg·K, ΔT in K → Q in kW
        delta_T = T_REGEN - T_inlet_c
        Q_sensible_kw = self.mea_flow_kgs * MEA_CP * delta_T

        # --- Step 2: Reaction heat — breaking CO₂-MEA bond ---
        # First find moles of CO₂ being stripped per second
        # CO₂ target flow: kg/h → kg/s
        co2_flow_kgs = self.co2_target_kgh / 3600.0

        # Moles of CO₂ per second [mol/s]
        co2_flow_mols = (co2_flow_kgs * 1000) / CO2_MOLAR_MASS

        # Heat of reaction [kW = kJ/s]
        Q_reaction_kw = co2_flow_mols * DELTA_H_ABS

        # --- Step 3: Vaporisation heat (water vapour in stripper overhead) ---
        # ~10–15% additional heat for water evaporation in reboiler
        # Using 12% as industry rule of thumb
        Q_vaporisation_kw = (Q_sensible_kw + Q_reaction_kw) * 0.12

        # --- Step 4: Total reboiler duty ---
        Q_total_kw = Q_sensible_kw + Q_reaction_kw + Q_vaporisation_kw

        # --- Step 5: Specific reboiler duty (benchmark metric) ---
        # Industry benchmark: 3.5–4.5 GJ/tonne CO₂
        # Our value should fall in this range for validation
        if self.co2_target_kgh > 0:
            # kW × 3600s = kJ/h; /1000 = MJ/h; co2 in kg/h /1000 = t/h
            specific_duty_gj_tonne = (Q_total_kw * 3.6) / (self.co2_target_kgh)
        else:
            specific_duty_gj_tonne = 0.0

        return {
            "T_inlet_c"                  : round(T_inlet_c, 2),
            "T_regen_c"                  : T_REGEN,
            "delta_T_c"                  : round(delta_T, 2),
            "Q_sensible_kw"              : round(Q_sensible_kw, 2),
            "Q_reaction_kw"              : round(Q_reaction_kw, 2),
            "Q_vaporisation_kw"          : round(Q_vaporisation_kw, 2),
            "Q_total_required_kw"        : round(Q_total_kw, 2),
            "specific_duty_gj_per_tonne" : round(specific_duty_gj_tonne, 2),
            "co2_available_kgh"          : round(self.co2_available_kgh, 2),
            "co2_target_kgh"             : round(self.co2_target_kgh, 2),
        }

    # ------------------------------------------------------------------
    def compute(self, hx_result: dict) -> dict:
        """
        Main method: given heat exchanger output, compute CCUS performance.

        Integrates with HX module output to determine:
        - How much heat waste recovery provides (Stage 1)
        - How much solar thermal is still needed (Stage 2)
        - Actual CO₂ captured based on available heat
        - Carbon credits generated

        Parameters
        ----------
        hx_result : dict
            Output from HeatExchanger.rate() — contains Q_delivered_ccus_kw
            and T_glycol_out_c.

        Returns
        -------
        dict with full CCUS performance metrics.
        """

        # --- Extract HX outputs ---
        Q_waste_heat_kw  = hx_result["Q_delivered_ccus_kw"]   # from glycol loop
        T_glycol_out     = hx_result["T_glycol_out_c"]         # pre-heat temp achieved

        # --- Heat requirement at the glycol pre-heat temperature ---
        heat_req = self.compute_heat_requirement(T_inlet_c=T_glycol_out)
        Q_total_required = heat_req["Q_total_required_kw"]

        # --- Stage 1: Waste heat contribution ---
        # Pre-heats MEA from T_ambient → T_glycol_out
        # Fraction of total sensible heat covered by waste heat
        sensible_total = heat_req["Q_sensible_kw"] + heat_req["Q_vaporisation_kw"] * \
                         (heat_req["Q_sensible_kw"] /
                         (heat_req["Q_sensible_kw"] + heat_req["Q_reaction_kw"]))

        # Direct waste heat fraction based on temperature lift
        T_lift_waste  = T_glycol_out - T_AMBIENT          # °C lift from waste heat
        T_lift_total  = T_REGEN     - T_AMBIENT           # °C total lift needed

        Q_waste_contribution_kw = min(
            Q_waste_heat_kw,
            Q_total_required * (T_lift_waste / T_lift_total)
        )
        Q_waste_contribution_kw = max(Q_waste_contribution_kw, 0.0)

        # --- Stage 2: Solar thermal deficit ---
        Q_solar_needed_kw = Q_total_required - Q_waste_contribution_kw

        # --- Actual CO₂ captured ---
        # If we have full heat → full capture rate
        # If heat is short → partial capture proportional to heat available
        # (In practice solar covers the deficit — optimizer decides how much)
        total_heat_available = Q_waste_contribution_kw + Q_solar_needed_kw
        # Assuming solar fully covers deficit (optimizer will vary this)
        capture_fraction = min(total_heat_available / Q_total_required, 1.0) \
                           * self.target_capture_rate

        co2_captured_kgh  = self.co2_available_kgh * capture_fraction
        co2_captured_kgs  = co2_captured_kgh / 3600.0

        # --- Carbon credits ---
        # 1 tonne = 1000 kg; credits per hour
        co2_captured_th   = co2_captured_kgh / 1000.0        # tonnes/hour
        credits_per_hour  = co2_captured_th * self.carbon_price_usd  # USD/hour

        # --- Waste heat offset value ---
        # How much did waste heat save in solar thermal cost?
        # Solar thermal cost ~0.04 USD/kWh (IRENA 2023)
        SOLAR_THERMAL_COST = 0.04   # USD/kWh
        waste_heat_savings_usd = Q_waste_contribution_kw * SOLAR_THERMAL_COST

        # --- Energy metrics ---
        waste_heat_pct = (Q_waste_contribution_kw / Q_total_required * 100) \
                          if Q_total_required > 0 else 0

        return {
            # Heat supply breakdown
            "Q_total_required_kw"        : round(Q_total_required, 2),
            "Q_waste_contribution_kw"    : round(Q_waste_contribution_kw, 2),
            "Q_solar_needed_kw"          : round(Q_solar_needed_kw, 2),
            "waste_heat_coverage_pct"    : round(waste_heat_pct, 1),

            # Temperatures
            "T_mea_inlet_c"              : round(T_glycol_out, 2),
            "T_regen_c"                  : T_REGEN,

            # CO₂ performance
            "co2_available_kgh"          : round(self.co2_available_kgh, 2),
            "co2_captured_kgh"           : round(co2_captured_kgh, 2),
            "co2_captured_kgs"           : round(co2_captured_kgs, 4),
            "capture_fraction_pct"       : round(capture_fraction * 100, 1),

            # Economics
            "carbon_credits_usd_per_hour": round(credits_per_hour, 4),
            "waste_heat_savings_usd_hr"  : round(waste_heat_savings_usd, 4),

            # Benchmark
            "specific_duty_gj_per_tonne" : heat_req["specific_duty_gj_per_tonne"],
        }

    # ------------------------------------------------------------------
    def simulate_day(self, hx_daily: list) -> list:
        """
        Simulate 24 hours given list of hourly HX results.
        """
        return [self.compute(hx_r) for hx_r in hx_daily]


# ---------------------------------------------------------------------------
# QUICK TEST
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from core.datacenter_thermal import DataCenterThermalModel, get_cpu_profile
    from core.heat_exchanger      import HeatExchanger

    print("=" * 65)
    print("  MEA REGENERATOR MODEL — QUICK TEST")
    print("=" * 65)

    # --- Setup pipeline ---
    dc  = DataCenterThermalModel()
    hx  = HeatExchanger(area_m2=25.0, glycol_flow_kgs=0.8, glycol_inlet_temp_c=25.0)
    mea = MEARegenerator(mea_flow_kgs=0.21, target_capture_rate=0.90)

    print(f"\n  CO₂ available in flue gas  : {mea.co2_available_kgh:.2f} kg/h")
    print(f"  CO₂ target (90% capture)   : {mea.co2_target_kgh:.2f} kg/h")

    # --- Full heat requirement (baseline — no pre-heat) ---
    print("\n📊 Heat Requirement Analysis")
    print("-" * 50)
    baseline = mea.compute_heat_requirement(T_inlet_c=25.0)
    print(f"  Sensible heat (25→120°C)   : {baseline['Q_sensible_kw']:>8.2f} kW")
    print(f"  Reaction heat (CO₂ strip)  : {baseline['Q_reaction_kw']:>8.2f} kW")
    print(f"  Vaporisation heat          : {baseline['Q_vaporisation_kw']:>8.2f} kW")
    print(f"  ─────────────────────────────────────")
    print(f"  TOTAL reboiler duty        : {baseline['Q_total_required_kw']:>8.2f} kW")
    print(f"  Specific duty              : {baseline['specific_duty_gj_per_tonne']:>8.2f} GJ/tonne CO₂")
    print(f"  (Industry benchmark: 3.5–4.5 GJ/tonne ← should be in this range)")

    # --- Single point: 80% CPU ---
    print("\n📊 Single Point: 80% CPU Load (Full Pipeline)")
    print("-" * 50)
    dc_out  = dc.compute(0.80)
    hx_out  = hx.rate(dc_out)
    mea_out = mea.compute(hx_out)

    print(f"  Heat required (total)      : {mea_out['Q_total_required_kw']:>8.2f} kW")
    print(f"  Waste heat contribution    : {mea_out['Q_waste_contribution_kw']:>8.2f} kW")
    print(f"  Solar thermal still needed : {mea_out['Q_solar_needed_kw']:>8.2f} kW")
    print(f"  Waste heat coverage        : {mea_out['waste_heat_coverage_pct']:>8.1f} %")
    print(f"  ─────────────────────────────────────")
    print(f"  CO₂ captured               : {mea_out['co2_captured_kgh']:>8.2f} kg/h")
    print(f"  Capture efficiency         : {mea_out['capture_fraction_pct']:>8.1f} %")
    print(f"  Carbon credits             : ${mea_out['carbon_credits_usd_per_hour']:>7.4f}/hour")

    # --- 24-hour simulation ---
    print("\n📅 24-Hour CCUS Performance (Office Profile)")
    print("-" * 78)
    profile  = get_cpu_profile("office")
    daily_dc = dc.simulate_day(profile)
    daily_hx = [hx.rate(r) for r in daily_dc]
    daily_mea= mea.simulate_day(daily_hx)

    print(f"  {'Hr':>3} {'CPU%':>6} {'Q_waste':>9} {'Q_solar':>9} {'WH_cov%':>8} {'CO2_kg/h':>10} {'Credits$':>10}")
    print(f"  {'─'*3} {'─'*6} {'─'*9} {'─'*9} {'─'*8} {'─'*10} {'─'*10}")

    total_co2   = 0
    total_creds = 0
    total_solar = 0
    total_waste = 0

    for hr, (dc_r, mea_r) in enumerate(zip(daily_dc, daily_mea)):
        total_co2   += mea_r["co2_captured_kgh"]
        total_creds += mea_r["carbon_credits_usd_per_hour"]
        total_solar += mea_r["Q_solar_needed_kw"]
        total_waste += mea_r["Q_waste_contribution_kw"]
        print(
            f"  {hr:02d}:00"
            f"  {dc_r['cpu_utilization_pct']:>5.1f}%"
            f"  {mea_r['Q_waste_contribution_kw']:>8.1f}kW"
            f"  {mea_r['Q_solar_needed_kw']:>8.1f}kW"
            f"  {mea_r['waste_heat_coverage_pct']:>7.1f}%"
            f"  {mea_r['co2_captured_kgh']:>9.2f}kg/h"
            f"  ${mea_r['carbon_credits_usd_per_hour']:>9.4f}"
        )

    print(f"\n  {'─'*65}")
    print(f"  Total CO₂ captured (24h)       : {total_co2:.2f}  kg")
    print(f"  Total carbon credits (24h)     : ${total_creds:.4f}")
    print(f"  Total waste heat used (24h)    : {total_waste:.1f} kWh")
    print(f"  Total solar thermal needed     : {total_solar:.1f} kWh")
    avg_wh_cov = (total_waste / (total_waste + total_solar) * 100) if (total_waste+total_solar) > 0 else 0
    print(f"  Avg waste heat coverage        : {avg_wh_cov:.1f}%")
    print(f"\n  ✅ Module 3 complete — ready to build solar_battery.py")