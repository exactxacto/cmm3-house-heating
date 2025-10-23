import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Load R-values from file
r_val = np.loadtxt('thermal_resistance/out/R_results_Expanded Polystyrene (EPS)_mineral_wool_Polyurethane (PUR).txt', usecols=(3,), skiprows=1)

# Parameters
width_house = 9.0  # meters
length_house = 4.0 # meters
height_house = 3.0 # meters


# Calculations
vol_house = width_house * length_house * height_house  # m^3
density_air = 1.225  # kg/m^3
mass_air = vol_house * density_air  # kg
C = mass_air * 1005  # J/K

print C

T0 = 25   # initial temperature

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
    return sol

# Plot results for each R
plt.figure()
for i, R in enumerate(r_val):
    sol = simulate(R)
    plt.plot(sol.t / 3600, sol.y[0], label=f"Room Temp (R={R:.2f})")

# Plot outside temperature
plt.plot(sol.t / 3600, np.sin(sol.t / (3600 * 12)) * 10 + 20, 'k--', label="Outside Temp")
plt.xlabel("Time [hours]")
plt.ylabel("Temperature [°C]")
plt.title("Room Temperature vs Time over 28 Days")
plt.legend()
plt.grid(True)
plt.show()
