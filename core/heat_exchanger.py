"""
=============================================================================
MODULE 2: Heat Exchanger — Glycol-Water Loop (LMTD Method)
=============================================================================
Project : GX×DX Closed-Loop Digital Twin
Author  : V Niranjana | IIT Jodhpur
Purpose : Models a counterflow shell-and-tube heat exchanger that transfers
          waste heat from server exhaust air into a glycol-water secondary
          loop, which then pre-heats the MEA solvent in the CCUS unit.

System Layout (counterflow):
  HOT SIDE  : Server exhaust air   → enters hot, exits cooler
  COLD SIDE : Glycol-water mixture → enters cool, exits warmer

Key Physics:
  Q    = U · A · ΔT_lm          (heat exchanger fundamental equation)
  ΔT_lm = LMTD for counterflow  (Log Mean Temperature Difference)
  Q    = ṁ · Cp · ΔT            (energy balance on each fluid stream)

Two modes:
  1. SIZING mode   — given Q required, find area A needed
  2. RATING mode   — given area A, find Q and outlet temperatures
=============================================================================
"""

import numpy as np

# ---------------------------------------------------------------------------
# FLUID PROPERTIES
# ---------------------------------------------------------------------------

# Glycol-water mixture (40% ethylene glycol by volume — standard HVAC mix)
# Good to -15°C, safe for data center cooling loops
GLYCOL_CP       = 3.500    # kJ/kg·K
GLYCOL_DENSITY  = 1058.0   # kg/m³

# Server exhaust air properties
AIR_CP          = 1.006    # kJ/kg·K
AIR_DENSITY     = 1.2      # kg/m³

# Overall Heat Transfer Coefficient U for air-to-liquid HX
# Air-side is the limiting resistance (low conductivity)
# Typical range for forced-air to liquid: 0.05 – 0.15 kW/m²·K
U_OVERALL       = 0.08     # kW/m²·K  (conservative, air-to-liquid)

# ---------------------------------------------------------------------------
# CORE CLASS
# ---------------------------------------------------------------------------

class HeatExchanger:
    """
    Counterflow shell-and-tube heat exchanger model using LMTD method.

    The hot side is server exhaust air.
    The cold side is glycol-water loop fluid.

    Parameters
    ----------
    area_m2 : float
        Heat transfer area [m²]. Default = 25 m² (sized for 100kW DC).
    u_overall : float
        Overall heat transfer coefficient [kW/m²·K]. Default = 0.08.
    glycol_flow_kgs : float
        Mass flow rate of glycol-water on cold side [kg/s]. Default = 3.0.
    glycol_inlet_temp_c : float
        Inlet temperature of glycol (returning from MEA unit) [°C].
        Default = 35°C (assumed return temperature from CCUS pre-heater).
    effectiveness_limit : float
        Max heat exchanger effectiveness (accounts for real losses). 
        Default = 0.85.
    """

    def __init__(
        self,
        area_m2             = 25.0,
        u_overall           = U_OVERALL,
        glycol_flow_kgs     = 0.8,
        glycol_inlet_temp_c = 25.0,
        effectiveness_limit = 0.85
    ):
        self.area_m2             = area_m2
        self.u_overall           = u_overall
        self.glycol_flow_kgs     = glycol_flow_kgs
        self.glycol_inlet_temp_c = glycol_inlet_temp_c
        self.effectiveness_limit = effectiveness_limit

    # ------------------------------------------------------------------
    def compute_lmtd(
        self,
        T_hot_in:  float,
        T_hot_out: float,
        T_cold_in: float,
        T_cold_out:float
    ) -> float:
        """
        Compute Log Mean Temperature Difference for a counterflow HX.

        Counterflow arrangement:
          Hot fluid  : T_hot_in  → T_hot_out   (left to right)
          Cold fluid : T_cold_out← T_cold_in   (right to left, counterflow)

        ΔT₁ = T_hot_in  - T_cold_out   (at one end)
        ΔT₂ = T_hot_out - T_cold_in    (at other end)
        ΔT_lm = (ΔT₁ - ΔT₂) / ln(ΔT₁/ΔT₂)

        Special case: if ΔT₁ == ΔT₂, LMTD = ΔT₁ (L'Hôpital's rule)
        """
        dT1 = T_hot_in  - T_cold_out   # temp difference at hot inlet end
        dT2 = T_hot_out - T_cold_in    # temp difference at hot outlet end

        # Guard against non-physical inputs (hot must be hotter than cold)
        if dT1 <= 0 or dT2 <= 0:
            return 0.0  # No heat transfer possible

        # Guard against log(0) when dT1 == dT2
        if abs(dT1 - dT2) < 1e-6:
            return dT1  # L'Hôpital limit: LMTD → ΔT when ΔT₁ = ΔT₂

        lmtd = (dT1 - dT2) / np.log(dT1 / dT2)
        return lmtd

    # ------------------------------------------------------------------
    def rate(
        self,
        dc_result: dict,
        air_flow_m3s: float = 2.5
    ) -> dict:
        """
        RATING MODE: Given DC thermal output and HX geometry,
        compute actual heat transferred and all outlet temperatures.

        This is the primary method called by the optimizer.

        Parameters
        ----------
        dc_result : dict
            Output from DataCenterThermalModel.compute() — contains
            outlet_temp_c, q_recoverable_kw, etc.
        air_flow_m3s : float
            Volumetric airflow rate from DC [m³/s].

        Returns
        -------
        dict with heat transfer results and temperatures.
        """

        # --- Extract inputs from DC module output ---
        T_air_in      = dc_result["outlet_temp_c"]       # Hot side inlet [°C]
        q_available   = dc_result["q_recoverable_kw"]    # Max available [kW]
        T_glycol_in   = self.glycol_inlet_temp_c         # Cold side inlet [°C]

        # --- Mass flow rates ---
        air_mass_flow = AIR_DENSITY * air_flow_m3s       # kg/s

        # --- Capacity rates (ṁ·Cp) for each fluid ---
        # These determine which fluid undergoes the larger temperature change
        C_air    = air_mass_flow        * AIR_CP          # kW/K
        C_glycol = self.glycol_flow_kgs * GLYCOL_CP       # kW/K
        C_min    = min(C_air, C_glycol)                   # limits heat transfer
        C_max    = max(C_air, C_glycol)

        # --- NTU (Number of Transfer Units) ---
        # NTU = U·A / C_min  — dimensionless measure of HX size
        NTU = (self.u_overall * self.area_m2) / C_min

        # --- Effectiveness using NTU-effectiveness method (counterflow) ---
        # ε = [1 - exp(-NTU(1-C*))] / [1 - C*·exp(-NTU(1-C*))]
        # where C* = C_min/C_max
        C_star = C_min / C_max

        if abs(1.0 - C_star) < 1e-6:
            # Special case: C* = 1 (balanced flow)
            effectiveness = NTU / (1.0 + NTU)
        else:
            exp_term      = np.exp(-NTU * (1.0 - C_star))
            effectiveness = (1.0 - exp_term) / (1.0 - C_star * exp_term)

        # Cap at physical limit
        effectiveness = min(effectiveness, self.effectiveness_limit)

        # --- Maximum possible heat transfer ---
        # Q_max = C_min × (T_hot_in - T_cold_in)
        Q_max = C_min * (T_air_in - T_glycol_in)

        # Can't transfer more than what's available from DC
        Q_max = min(Q_max, q_available)

        # --- Actual heat transferred ---
        Q_actual = effectiveness * Q_max
        Q_actual = max(Q_actual, 0.0)  # non-negative

        # --- Outlet temperatures ---
        # Energy balance: Q = C_fluid × ΔT
        T_air_out    = T_air_in    - Q_actual / C_air     # Air cools down
        T_glycol_out = T_glycol_in + Q_actual / C_glycol  # Glycol heats up

        # --- LMTD (verification — should be consistent with Q = U·A·LMTD) ---
        lmtd = self.compute_lmtd(T_air_in, T_air_out, T_glycol_in, T_glycol_out)
        Q_check = self.u_overall * self.area_m2 * lmtd   # Should ≈ Q_actual

        # --- Glycol loop thermal energy delivered to CCUS pre-heater ---
        # This is what the MEA regenerator module will receive as input
        q_delivered_to_ccus = Q_actual

        # --- Heat recovery efficiency (actual vs available) ---
        recovery_actual = (Q_actual / q_available * 100) if q_available > 0 else 0

        return {
            # Temperatures
            "T_air_in_c"           : round(T_air_in, 2),
            "T_air_out_c"          : round(T_air_out, 2),
            "T_glycol_in_c"        : round(T_glycol_in, 2),
            "T_glycol_out_c"       : round(T_glycol_out, 2),

            # Heat transfer
            "Q_actual_kw"          : round(Q_actual, 2),
            "Q_max_kw"             : round(Q_max, 2),
            "Q_delivered_ccus_kw"  : round(q_delivered_to_ccus, 2),
            "Q_check_lmtd_kw"      : round(Q_check, 2),   # should ≈ Q_actual

            # HX performance
            "effectiveness_pct"    : round(effectiveness * 100, 1),
            "lmtd_k"               : round(lmtd, 2),
            "ntu"                  : round(NTU, 3),
            "recovery_actual_pct"  : round(recovery_actual, 1),

            # Capacity rates (useful for dashboard display)
            "C_air_kw_k"           : round(C_air, 3),
            "C_glycol_kw_k"        : round(C_glycol, 3),
            "C_star"               : round(C_star, 3),
        }

    # ------------------------------------------------------------------
    def size_for_duty(
        self,
        Q_required_kw: float,
        T_hot_in_c:    float,
        T_hot_out_c:   float,
        T_cold_in_c:   float,
        T_cold_out_c:  float
    ) -> dict:
        """
        SIZING MODE: Given required heat duty and terminal temperatures,
        compute required heat exchanger area.

        Useful for design phase: "How big does the HX need to be?"

        Returns
        -------
        dict with required area and LMTD.
        """
        lmtd     = self.compute_lmtd(T_hot_in_c, T_hot_out_c, T_cold_in_c, T_cold_out_c)

        if lmtd <= 0:
            return {"error": "Non-physical temperatures — check inputs"}

        # Q = U · A · LMTD  →  A = Q / (U · LMTD)
        area_required = Q_required_kw / (self.u_overall * lmtd)

        return {
            "Q_required_kw"  : round(Q_required_kw, 2),
            "lmtd_k"         : round(lmtd, 2),
            "u_overall"      : self.u_overall,
            "area_required_m2": round(area_required, 2),
            "T_hot_in_c"     : T_hot_in_c,
            "T_hot_out_c"    : T_hot_out_c,
            "T_cold_in_c"    : T_cold_in_c,
            "T_cold_out_c"   : T_cold_out_c,
        }


# ---------------------------------------------------------------------------
# QUICK TEST
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    # Import Day 1 module
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from core.datacenter_thermal import DataCenterThermalModel, get_cpu_profile

    print("=" * 60)
    print("  HEAT EXCHANGER MODEL — QUICK TEST")
    print("=" * 60)

    # --- Setup ---
    dc  = DataCenterThermalModel()
    hx  = HeatExchanger(
    area_m2             = 25.0,
    glycol_flow_kgs     = 0.8,
    glycol_inlet_temp_c = 25.0
)

    # --- Single point: 80% CPU ---
    print("\n📊 Single Point: 80% CPU Load")
    print("-" * 40)
    dc_out = dc.compute(0.80)
    hx_out = hx.rate(dc_out)

    print(f"  Air in          : {hx_out['T_air_in_c']} °C")
    print(f"  Air out         : {hx_out['T_air_out_c']} °C")
    print(f"  Glycol in       : {hx_out['T_glycol_in_c']} °C")
    print(f"  Glycol out      : {hx_out['T_glycol_out_c']} °C")
    print(f"  LMTD            : {hx_out['lmtd_k']} K")
    print(f"  NTU             : {hx_out['ntu']}")
    print(f"  Effectiveness   : {hx_out['effectiveness_pct']} %")
    print(f"  Q_actual        : {hx_out['Q_actual_kw']} kW")
    print(f"  Q_check (LMTD)  : {hx_out['Q_check_lmtd_kw']} kW  ← should ≈ Q_actual")
    print(f"  Q to CCUS       : {hx_out['Q_delivered_ccus_kw']} kW")

    # --- 24-hour simulation ---
    print("\n\n📅 24-Hour HX Performance (Office Profile)")
    print("-" * 70)
    profile = get_cpu_profile("office")
    daily_dc = dc.simulate_day(profile)

    print(f"  {'Hour':<6} {'CPU%':<7} {'Air In':>8} {'Air Out':>9} {'Glycol Out':>11} {'LMTD':>7} {'Q_kW':>8} {'Eff%':>7}")
    print(f"  {'-'*6} {'-'*6} {'-'*8} {'-'*8} {'-'*10} {'-'*7} {'-'*7} {'-'*6}")

    total_Q = 0
    for hour, dc_r in enumerate(daily_dc):
        hx_r = hx.rate(dc_r)
        total_Q += hx_r["Q_actual_kw"]
        print(
            f"  {hour:02d}:00  "
            f"  {dc_r['cpu_utilization_pct']:>5.1f}%"
            f"  {hx_r['T_air_in_c']:>8.1f}°C"
            f"  {hx_r['T_air_out_c']:>8.1f}°C"
            f"  {hx_r['T_glycol_out_c']:>9.1f}°C"
            f"  {hx_r['lmtd_k']:>7.1f}K"
            f"  {hx_r['Q_actual_kw']:>7.1f}kW"
            f"  {hx_r['effectiveness_pct']:>6.1f}%"
        )

    print(f"\n  Total heat delivered to CCUS (24h) : {total_Q:.1f} kWh")

    # --- Sizing example ---
    print("\n\n🔧 Sizing Mode: How big does HX need to be for 80kW duty?")
    print("-" * 50)
    sizing = hx.size_for_duty(
        Q_required_kw = 20.0,
        T_hot_in_c    = 48.0,
        T_hot_out_c   = 38.0,
        T_cold_in_c   = 25.0,
        T_cold_out_c  = 45.0
    )
    for k, v in sizing.items():
        print(f"  {k:<25} {v}")

    print(f"\n  ✅ Module 2 complete — ready to pipe into mea_regenerator.py")