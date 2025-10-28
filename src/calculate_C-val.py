# Parameters
from matplotlib import pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

width_house = 9.0  # meters
length_house = 4.0 # meters
height_house = 3.0 # meters


# Calculations
vol_house = width_house * length_house * height_house  # m^3
density_air = 1.225  # kg/m^3
mass_air = vol_house * density_air  # kg
C = mass_air * 1005  # J/K
print (C)


R = 5.194


def simulate(R):
    # Define the ODE for a given R

    def dTdt(t, T):
        T_out = np.sin(t / (3600 * 12)) * 10 + 20  # one cycle every 24 hours
        return (T_out - T) / (R * C)
    
    # Solve the ODE for 28 days (in seconds)
    t_span = (0, 28 * 24 * 3600)  # 28 days
    t_eval = np.linspace(t_span[0], t_span[1], 28 * 24 + 1)  # one point per hour
    sol = solve_ivp(dTdt, t_span, [T0], t_eval=t_eval)
    print(sol.t)

#plt.plot(sol.t / 3600, np.sin(sol.t / (3600 * 12)) * 10 + 20, 'k--', label="Outside Temp")