"""
thermal_comfort_model_REALDATA_v2.py

PURPOSE:
    Simulate indoor thermal comfort using real or synthetic hourly outdoor
    temperature data (Edinburgh 2023). The model uses a lumped thermal mass
    ODE with thermostat-controlled heating. It finds the insulation thickness
    that maximizes comfort hours and minimizes heating energy.

REQUIREMENTS:
    pip install numpy pandas matplotlib
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, Optional


# ---------------------------------------------------------
# 1. THERMAL PHYSICS FUNCTIONS
# ---------------------------------------------------------
def compute_R_total(insulation_thickness_m: float,
                    k_insulation: float,
                    base_R_other_layers: float = 0.25) -> float:
    """
    Compute total wall thermal resistance (R-value, m²·K/W):
        R_total = R_base + thickness / k
    """
    R_ins = insulation_thickness_m / k_insulation
    return base_R_other_layers + R_ins


def simulate_lumped_model(T_out_series: pd.Series,
                          insulation_thickness_m: float,
                          area_m2: float,
                          k_insulation: float,
                          C_total: float,
                          T_init: Optional[float] = None,
                          base_R_other_layers: float = 0.25,
                          comfort_lower: float = 18.0,
                          comfort_upper: float = 24.0,
                          heater_power_W: float = 2000.0) -> Tuple[pd.Series, float, int]:
    """
    Simulates indoor temperature using 1st-order transient model:
        C * dT_in/dt = (A/R_total)*(T_out - T_in) + Q_heating

    Heating logic:
        - Heater ON if T_in < comfort_lower (18°C)
        - Heater OFF otherwise

    Returns:
        - Indoor temperature series
        - Total heating energy used [kWh]
        - Total heating-on hours [h]
    """

    times = T_out_series.index
    dt_hours = (times[1] - times[0]).total_seconds() / 3600.0
    dt = dt_hours * 3600.0  # seconds per time step

    R_per_area = compute_R_total(insulation_thickness_m, k_insulation, base_R_other_layers)
    coeff = area_m2 / (R_per_area * C_total)  # 1/s

    # Initial condition
    T0 = T_out_series.iloc[0] if T_init is None else T_init
    T_in = np.zeros(len(T_out_series))
    T_in[0] = T0

    heater_on_hours = 0
    heater_energy_Wh = 0.0

    for i in range(1, len(T_out_series)):
        T_prev = T_in[i - 1]
        Tout_prev = T_out_series.iloc[i - 1]

        # Thermostat control
        Q_heating = 0.0
        if T_prev < comfort_lower:
            Q_heating = heater_power_W  # ON
            heater_on_hours += 1
            heater_energy_Wh += heater_power_W * dt_hours

        dTdt = coeff * (Tout_prev - T_prev) + (Q_heating / C_total)
        T_in[i] = T_prev + dTdt * dt

    total_energy_kWh = heater_energy_Wh / 1000.0
    return pd.Series(T_in, index=times), total_energy_kWh, heater_on_hours


# ---------------------------------------------------------
# 2. COMFORT METRICS + ITERATIVE OPTIMIZATION
# ---------------------------------------------------------
def comfort_metrics(T_in_series: pd.Series,
                    comfort_lower: float = 18.0,
                    comfort_upper: float = 24.0) -> dict:
    """Calculate hours within and outside the comfort band."""
    total = len(T_in_series)
    comfy = ((T_in_series >= comfort_lower) & (T_in_series <= comfort_upper)).sum()
    cold = (T_in_series < comfort_lower).sum()
    hot = (T_in_series > comfort_upper).sum()
    return {
        "total_hours": int(total),
        "comfortable_hours": int(comfy),
        "heating_hours": int(cold),
        "cooling_hours": int(hot),
        "discomfort_fraction": 1.0 - comfy / total,
    }


def optimize_insulation_iterative(T_out_series: pd.Series,
                                  area_m2: float,
                                  k_insulation: float,
                                  C_total: float,
                                  base_R_other_layers: float = 0.25,
                                  comfort_bounds: Tuple[float, float] = (18.0, 24.0),
                                  thickness_range=np.arange(0.00, 0.41, 0.02)) -> dict:
    """
    Iterates over insulation thickness values to find the one giving
    the maximum comfort hours (and record heating energy).
    """
    results = []

    for thickness in thickness_range:
        T_in, energy_kWh, hours_on = simulate_lumped_model(
            T_out_series, thickness, area_m2, k_insulation, C_total,
            base_R_other_layers=base_R_other_layers,
            comfort_lower=comfort_bounds[0], comfort_upper=comfort_bounds[1]
        )
        metrics = comfort_metrics(T_in, comfort_lower=comfort_bounds[0], comfort_upper=comfort_bounds[1])
        metrics['thickness_m'] = thickness
        metrics['energy_kWh'] = energy_kWh
        metrics['heater_hours'] = hours_on
        metrics['T_in'] = T_in
        results.append(metrics)

    df_results = pd.DataFrame(results)
    idx_best = df_results['comfortable_hours'].idxmax()
    best = df_results.iloc[idx_best]

    print(f"\n✅ Best insulation thickness: {best['thickness_m']:.3f} m")
    print(f"Comfortable hours: {best['comfortable_hours']} of {len(T_out_series)}")
    print(f"Heating energy: {best['energy_kWh']:.1f} kWh")
    print(f"Heating ON hours: {best['heater_hours']}")

    plt.figure(figsize=(8, 4))
    plt.plot(df_results['thickness_m'], df_results['comfortable_hours'], '-o', color='tab:green')
    plt.xlabel("Insulation Thickness (m)")
    plt.ylabel("Comfortable Hours per Year")
    plt.title("Comfort vs. Insulation Thickness")
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(8, 4))
    plt.plot(df_results['thickness_m'], df_results['energy_kWh'], '-o', color='tab:red')
    plt.xlabel("Insulation Thickness (m)")
    plt.ylabel("Heating Energy (kWh)")
    plt.title("Heating Energy vs. Insulation Thickness")
    plt.grid(True)
    plt.show()

    return {
        'best_thickness_m': best['thickness_m'],
        'metrics': best,
        'all_results': df_results,
        'T_in_series': best['T_in']
    }


# ---------------------------------------------------------
# 3. PLOTTING
# ---------------------------------------------------------
def plot_temperature_series(T_out: pd.Series,
                            T_in: pd.Series,
                            comfort_lower: float = 18.0,
                            comfort_upper: float = 24.0,
                            title: str = "Indoor vs Outdoor Temperature"):
    """Plot indoor/outdoor temperature and comfort band."""
    plt.figure(figsize=(15, 4))
    plt.plot(T_out.index, T_out.values, label="Outdoor Temp (°C)", color="tab:blue", alpha=0.7)
    plt.plot(T_in.index, T_in.values, label="Indoor Temp (°C)", color="tab:orange")
    plt.fill_between(T_in.index, comfort_lower, comfort_upper,
                     color="green", alpha=0.2, label="Comfort band (18–24°C)")
    plt.xlabel("Date")
    plt.ylabel("Temperature (°C)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------
# 4. MAIN FUNCTION WITH REAL OR SYNTHETIC DATA
# ---------------------------------------------------------
def example_with_real_or_synthetic_data():
    """
    Load Edinburgh 2023 hourly temperature data from CSV if available,
    else generate synthetic data.
    """
    import os

    csv_path = "C:/Users/yourname/Documents/edinburgh_2023_hourly_temps.csv"  # CHANGE THIS

    if os.path.exists(csv_path):
        print("✅ Using real temperature data from CSV...")
        df = pd.read_csv(csv_path)
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.set_index("datetime")
        T_out_series = df["T_out"].asfreq("h").interpolate()
    else:
        print("⚠️ CSV file not found — generating synthetic Edinburgh-like temperature data...")
        rng = pd.date_range("2023-01-01", "2023-12-31 23:00", freq="h")
        day = rng.dayofyear.values
        T_out = 8 + 7 * np.sin(2 * np.pi * (day - 170) / 365) + 3 * np.random.randn(len(rng))
        T_out_series = pd.Series(T_out, index=rng)
        print(f"✅ Synthetic data created for 2023 ({len(T_out_series)} hourly points).")

    # Parameters
    area = 50.0
    k_ins = 0.04
    C_tot = 1.5e6
    base_R = 0.25

    opt = optimize_insulation_iterative(T_out_series, area, k_ins, C_tot, base_R,
                                        comfort_bounds=(18, 24))
    best_t = opt["best_thickness_m"]
    metrics = opt["metrics"]

    print("\n--- FINAL RESULTS ---")
    print(f"Optimal insulation thickness: {best_t:.3f} m")
    print(f"Comfortable hours: {metrics['comfortable_hours']} of {metrics['total_hours']}")
    print(f"Heating ON hours: {metrics['heater_hours']}")
    print(f"Total heating energy: {metrics['energy_kWh']:.1f} kWh")

    plot_temperature_series(T_out_series, opt["T_in_series"],
                            comfort_lower=18, comfort_upper=24,
                            title=f"Indoor vs Outdoor (optimum thickness={best_t:.3f} m)")


# ---------------------------------------------------------
# RUN
# ---------------------------------------------------------
if __name__ == "__main__":
    example_with_real_or_synthetic_data()
    
    
Thermal_comfort_multilayer_cost.py

PURPOSE:
    Compare thermal comfort, heating demand, and material cost for different
    multi-layer insulation combinations (total thickness = 0.1 m)
    using a transient thermal model with thermostat heating.

REQUIREMENTS:
    pip install numpy pandas matplotlib
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional


# ---------------------------------------------------------
# 1️⃣ THERMAL RESISTANCE CALCULATION
# ---------------------------------------------------------
def compute_R_total_from_layers(layers: list[tuple[str, float]],
                                k_values: dict,
                                base_R_other_layers: float = 0.25) -> float:
    """
    Compute total wall thermal resistance (m²·K/W) from multiple insulation layers.
    layers: [(material_name, thickness_m), ...]
    k_values: dict mapping material_name -> conductivity [W/m·K]
    """
    R_total = base_R_other_layers
    for material, thickness in layers:
        k = k_values[material]
        R_total += thickness / k
    return R_total


# ---------------------------------------------------------
# 2️⃣ INDOOR TEMPERATURE SIMULATION (with heating control)
# ---------------------------------------------------------
def simulate_lumped_model_R(T_out_series: pd.Series,
                            R_total: float,
                            area_m2: float,
                            C_total: float,
                            T_init: Optional[float] = None,
                            comfort_lower: float = 18.0,
                            heater_power_W: float = 2000.0):
    """
    Simulates indoor temperature using:
        C * dT_in/dt = (A / R_total)*(T_out - T_in) + Q_heating
    Heating ON if T_in < comfort_lower.
    """
    times = T_out_series.index
    dt_hours = (times[1] - times[0]).total_seconds() / 3600.0
    dt = dt_hours * 3600.0  # seconds per step

    coeff = area_m2 / (R_total * C_total)
    T0 = T_out_series.iloc[0] if T_init is None else T_init
    T_in = np.zeros(len(T_out_series))
    T_in[0] = T0

    heater_on_hours = 0
    heater_energy_Wh = 0.0

    for i in range(1, len(T_out_series)):
        T_prev = T_in[i - 1]
        Tout_prev = T_out_series.iloc[i - 1]

        # Thermostat control
        Q_heating = 0.0
        if T_prev < comfort_lower:
            Q_heating = heater_power_W
            heater_on_hours += 1
            heater_energy_Wh += heater_power_W * dt_hours

        dTdt = coeff * (Tout_prev - T_prev) + Q_heating / C_total
        T_in[i] = T_prev + dTdt * dt

    total_energy_kWh = heater_energy_Wh / 1000.0
    return pd.Series(T_in, index=times), total_energy_kWh, heater_on_hours


# ---------------------------------------------------------
# 3️⃣ COMFORT METRICS
# ---------------------------------------------------------
def comfort_metrics(T_in_series: pd.Series,
                    comfort_lower: float = 18.0,
                    comfort_upper: float = 24.0) -> dict:
    """Calculate hours within and outside the comfort band."""
    total = len(T_in_series)
    comfy = ((T_in_series >= comfort_lower) & (T_in_series <= comfort_upper)).sum()
    cold = (T_in_series < comfort_lower).sum()
    hot = (T_in_series > comfort_upper).sum()
    return {
        "total_hours": int(total),
        "comfortable_hours": int(comfy),
        "heating_hours": int(cold),
        "cooling_hours": int(hot),
        "discomfort_fraction": 1.0 - comfy / total,
    }


# ---------------------------------------------------------
# 4️⃣ MAIN ANALYSIS FUNCTION WITH COST CALCULATION
# ---------------------------------------------------------
def compare_wall_combinations_with_cost(T_out_series: pd.Series):
    """
    Compare comfort, heating, and cost performance for multiple
    insulation layer combinations (total 0.10 m).
    """

    # Thermal conductivities [W/m·K]
    k_values = {
        "EPS": 0.035,
        "MW": 0.040,
        "PUR": 0.025,
        "XPS": 0.030,
    }

    # Material costs (£ per m² per m of thickness)
    cost_per_m2_per_m = {
        "EPS": 20.0,
        "MW": 12.0,
        "PUR": 50.0,
        "XPS": 30.0,
    }

    # Wall combinations (total 0.10 m)
    walls = {
        "EPS only": [("EPS", 0.10)],
        "Mineral wool only": [("MW", 0.10)],
        "PUR only": [("PUR", 0.10)],
        "XPS only": [("XPS", 0.10)],
        "EPS + PUR": [("EPS", 0.05), ("PUR", 0.05)],
        "MW + XPS": [("MW", 0.05), ("XPS", 0.05)],
        "EPS + MW + XPS": [("EPS", 0.03), ("MW", 0.04), ("XPS", 0.03)],
    }

    # Building constants
    area = 50.0
    C_tot = 1.5e6
    base_R = 0.25

    results = []

    for name, layers in walls.items():
        R_total = compute_R_total_from_layers(layers, k_values, base_R)
        T_in, energy_kWh, hours_on = simulate_lumped_model_R(
            T_out_series, R_total, area, C_tot
        )
        metrics = comfort_metrics(T_in)
        combo_cost = sum(thk * cost_per_m2_per_m[mat] for (mat, thk) in layers)

        metrics.update({
            "Wall type": name,
            "R_total": R_total,
            "U_value": 1 / R_total,
            "Energy_kWh": energy_kWh,
            "Heater_hours": hours_on,
            "Cost_per_m2": combo_cost,
            "T_in": T_in,
        })
        results.append(metrics)

        print(f"\n🏗 {name}")
        print(f"  R_total = {R_total:.3f} m²K/W  |  U = {1/R_total:.3f} W/m²K")
        print(f"  Comfort hours: {metrics['comfortable_hours']} / {metrics['total_hours']}")
        print(f"  Heating energy: {energy_kWh:.1f} kWh | Heating hours: {hours_on}")
        print(f"  Cost per m²: £{combo_cost:.2f}")

    df = pd.DataFrame(results)

    # --- PLOTS ---
    plt.figure(figsize=(8, 5))
    plt.bar(df["Wall type"], df["comfortable_hours"], color="seagreen")
    plt.ylabel("Comfortable Hours per Year")
    plt.xticks(rotation=30, ha="right")
    plt.title("Comfort Hours by Wall Type")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.bar(df["Wall type"], df["Energy_kWh"], color="firebrick")
    plt.ylabel("Heating Energy (kWh)")
    plt.xticks(rotation=30, ha="right")
    plt.title("Heating Energy by Wall Type")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.bar(df["Wall type"], df["Cost_per_m2"], color="royalblue")
    plt.ylabel("Material Cost (£/m²)")
    plt.xticks(rotation=30, ha="right")
    plt.title("Insulation Material Cost by Wall Type")
    plt.tight_layout()
    plt.show()

    return df


# ---------------------------------------------------------
# 5️⃣ REAL OR SYNTHETIC TEMPERATURE DATA
# ---------------------------------------------------------
def get_temperature_data():
    import os
    csv_path = "C:/Users/yourname/Documents/edinburgh_2023_hourly_temps.csv"

    if os.path.exists(csv_path):
        print("✅ Using real Edinburgh temperature data...")
        df = pd.read_csv(csv_path)
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.set_index("datetime")
        return df["T_out"].asfreq("h").interpolate()
    else:
        print("⚠️ CSV file not found — generating synthetic Edinburgh-like data...")
        rng = pd.date_range("2023-01-01", "2023-12-31 23:00", freq="h")
        day = rng.dayofyear.values
        T_out = 8 + 7 * np.sin(2 * np.pi * (day - 170) / 365) + 3 * np.random.randn(len(rng))
        print(f"✅ Synthetic dataset generated ({len(T_out)} hours).")
        return pd.Series(T_out, index=rng)


# ---------------------------------------------------------
# 6️⃣ RUN
# ---------------------------------------------------------
if __name__ == "__main__":
    T_out = get_temperature_data()
    df_results = compare_wall_combinations_with_cost(T_out)

    print("\n--- SUMMARY TABLE ---")
    print(df_results[["Wall type", "R_total", "U_value", "comfortable_hours",
                      "Energy_kWh", "Heater_hours", "Cost_per_m2"]])

