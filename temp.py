# -*- coding: utf-8 -*-
# thermal_comfort_model_with_plot.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, Optional
from scipy.optimize import minimize_scalar

# (Reuse compute_R_total, simulate_lumped_model, comfort_metrics, objective_insulation, optimize_insulation from earlier)
# I’ll re-include minimal needed here plus plotting.

def compute_R_total(insulation_thickness_m: float,
                    k_insulation: float,
                    base_R_other_layers: float = 0.2) -> float:
    """R per area (m2K/W)."""
    R_ins = insulation_thickness_m / k_insulation
    return base_R_other_layers + R_ins

def simulate_lumped_model(T_out_series: pd.Series,
                          insulation_thickness_m: float,
                          area_m2: float,
                          k_insulation: float,
                          C_total: float,
                          Q_heating_func=None,
                          T_init: Optional[float] = None,
                          base_R_other_layers: float = 0.2) -> pd.Series:
    times = T_out_series.index
    dt_hours = (times[1] - times[0]).total_seconds()/3600.0
    dt = dt_hours * 3600.0  # seconds

    R_per_area = compute_R_total(insulation_thickness_m, k_insulation, base_R_other_layers)
    # heat exchange coefficient factor (1/s)
    coeff = area_m2 / (R_per_area * C_total)

    if T_init is None:
        T0 = T_out_series.iloc[0]
    else:
        T0 = T_init

    vals_out = T_out_series.values
    N = len(vals_out)
    T_in = np.zeros(N)
    T_in[0] = T0

    def Q_heating(i, T_i, T_out_i):
        if Q_heating_func is None:
            return 0.0
        return Q_heating_func(i, T_i, T_out_i)

    for i in range(1, N):
        T_prev = T_in[i-1]
        Tout_prev = vals_out[i-1]
        q = Q_heating(i-1, T_prev, Tout_prev)
        dTdt = coeff * (Tout_prev - T_prev) + (q / C_total)
        T_in[i] = T_prev + dTdt * dt

    return pd.Series(T_in, index=times)

def plot_temperature_series(T_out: pd.Series,
                            T_in: pd.Series,
                            comfort_lower: float = 18.0,
                            comfort_upper: float = 24.0,
                            title: str = "Indoor vs Outdoor Temperature"):
    """
    Plot outdoor and indoor temperature, shading comfort band, 
    and mark periods of discomfort.
    """
    plt.figure(figsize=(15,4))
    plt.plot(T_out.index, T_out.values, label="Outdoor Temp (°C)", color='tab:blue', alpha=0.7)
    plt.plot(T_in.index, T_in.values, label="Indoor Temp (°C)", color='tab:orange')

    # shade comfort band
    plt.fill_between(T_in.index, comfort_lower, comfort_upper,
                     color='green', alpha=0.2, label=f"Comfort band {comfort_lower}–{comfort_upper} °C")

    # optionally highlight times when indoor < lower or > upper
    below = T_in < comfort_lower
    above = T_in > comfort_upper
    plt.scatter(T_in.index[below], T_in[below], color='blue', s=2, label="Too cold")
    plt.scatter(T_in.index[above], T_in[above], color='red', s=2, label="Too hot")

    plt.xlabel("Date / Time")
    plt.ylabel("Temperature (°C)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()

def example_with_plot_synthetic():
    # synthetic outdoor for year, like before
    rng = pd.date_range('2023-01-01', periods=24*365, freq='H')
    day = rng.dayofyear.values
    T_out = 8 + 7*np.sin(2*np.pi*(day - 170)/365) + 3*np.random.randn(len(rng))
    T_out_series = pd.Series(T_out, index=rng)

    # parameters
    area = 50.0
    k_ins = 0.04
    C_tot = 1.5e6
    base_R = 0.25

    # pick a thickness, or optimize
    opt = optimize_insulation(T_out_series, area, k_ins, C_tot, base_R,
                              thickness_bounds=(0.0, 0.4), comfort_bounds=(18,24))
    best_t = opt['best_thickness_m']
    print("Best thickness (m):", best_t)
    T_in_best = opt['T_in_series']
    metrics = opt['metrics']
    print("Comfort metrics:", metrics)

    # plot
    plot_temperature_series(T_out_series, T_in_best, comfort_lower=18, comfort_upper=24,
                            title=f"Indoor vs Outdoor, thickness={best_t:.3f} m")
    return opt

# To integrate: import the previously defined optimize_insulation
def optimize_insulation(T_out_series: pd.Series,
                        area_m2: float,
                        k_insulation: float,
                        C_total: float,
                        base_R_other_layers: float = 0.2,
                        thickness_bounds: Tuple[float,float]=(0.0, 0.4),
                        comfort_bounds: Tuple[float,float]=(18.0,24.0),
                        maximize_comfort: bool = True) -> dict:
    # objective as before
    def obj_fn(t):
        return objective_insulation(t, T_out_series, area_m2, k_insulation, C_total,
                                    base_R_other_layers=base_R_other_layers,
                                    comfort_bounds=comfort_bounds, maximize_comfort=maximize_comfort)
    res = minimize_scalar(obj_fn, bounds=thickness_bounds, method='bounded', options={'xatol':1e-3})
    best_t = res.x
    T_in = simulate_lumped_model(T_out_series, best_t, area_m2, k_insulation, C_total,
                                 Q_heating_func=None, T_init=None,
                                 base_R_other_layers=base_R_other_layers)
    metrics = comfort_metrics(T_in, comfort_lower=comfort_bounds[0], comfort_upper=comfort_bounds[1])
    return {
        'opt_result': res,
        'best_thickness_m': best_t,
        'metrics': metrics,
        'T_in_series': T_in
    }

# (reuse objective_insulation, comfort_metrics from prior code)

def objective_insulation(thickness_m: float,
                         T_out_series: pd.Series,
                         area_m2: float,
                         k_insulation: float,
                         C_total: float,
                         base_R_other_layers: float = 0.2,
                         comfort_bounds: Tuple[float,float] = (18.0, 24.0),
                         maximize_comfort: bool = True) -> float:
    if thickness_m < 0:
        return 1e6
    T_in = simulate_lumped_model(T_out_series, thickness_m, area_m2, k_insulation,
                                 C_total, Q_heating_func=None, T_init=None,
                                 base_R_other_layers=base_R_other_layers)
    m = comfort_metrics(T_in, comfort_lower=comfort_bounds[0], comfort_upper=comfort_bounds[1])
    if maximize_comfort:
        return -m['comfortable_hours']
    else:
        return m['discomfort_fraction']

def comfort_metrics(T_in_series: pd.Series,
                    comfort_lower: float = 18.0,
                    comfort_upper: float = 24.0) -> dict:
    total = len(T_in_series)
    comfy = ((T_in_series >= comfort_lower) & (T_in_series <= comfort_upper)).sum()
    cold = (T_in_series < comfort_lower).sum()
    hot = (T_in_series > comfort_upper).sum()
    return {
        'total_hours': int(total),
        'comfortable_hours': int(comfy),
        'heating_hours': int(cold),
        'cooling_hours': int(hot),
        'discomfort_fraction': 1.0 - comfy / total
    }

if __name__ == "__main__":
    example_with_plot_synthetic()
