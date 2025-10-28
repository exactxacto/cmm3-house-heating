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

T0 = 25   # initial temperature

def simulate(R_eff):
# Define the ODE for a given R
    def dTdt(t, T):
        T_out_eff = #load T_out_eff from simulate_solar_irradiance.py
        return ((T_out_eff - T) / (R_eff * C))

    # Solve the ODE for 28 days (in seconds)
    t_span = (0, 28 * 24 * 3600)  # 28 days
    t_eval = np.linspace(t_span[0], t_span[1], 28 * 24 + 1)  # one point per hour
    sol = solve_ivp(dTdt, t_span, [T0], t_eval=t_eval)
    print(sol.t)
    return sol

for i, R in enumerate(r_val):
    sol = simulate(R)