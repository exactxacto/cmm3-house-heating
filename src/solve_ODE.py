import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Parameters
width_house = 9.0  # meters
length_house = 4.0 # meters
height_house = 3.0 # meters

# Calculations
vol_house = width_house * length_house * height_house  # m^3
density_air = 1.225  # kg/m^3
mass_air = vol_house * density_air  # kg
C = mass_air * 1005  # J/K

T0 = 25   # initial temperature, replace with desired initial temperature

results = {}

def simulate(R_label):
    T_out_series = T_out_df[R_label].values       # hourly external temperatures
    R_eff = R_eff_dict[R_label]                   # total resistance for this wall
    
    def dTdt(t, T):
        # Convert simulation time (s) → hours for indexing
        hour = t / 3600
        # Linear interpolation between hourly values
        T_out_t = np.interp(hour, time_hours, T_out_series)
        return (T_out_t - T) / (R_eff * C)

     # One-year simulation (8760 hours)
    t_span = (0, n_hours * 3600)
    t_eval = np.linspace(t_span[0], t_span[1], n_hours)
    sol = solve_ivp(dTdt, t_span, [T0], t_eval=t_eval, method='RK45') #solves using Runge-Kutta method

    T_hourly = sol.y[0]           # shape (n_hours,)
    t_hourly = sol.t / 3600       # hours
    return t_hourly, T_hourly

output_rows = []
for R_label in columns:
    t_hourly, T_hourly = simulate(R_label)
    R_val = R_eff_dict[R_label]
    row_data = np.concatenate((np.array([R_val]), T_hourly))
    output_rows.append(row_data)

results_array = np.vstack(output_rows)

