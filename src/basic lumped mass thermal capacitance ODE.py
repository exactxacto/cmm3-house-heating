import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

r_val = np.loadtxt('', usecols=(COLUMN_INDEX,))
# Parameters
R = r_val     # K/W
C = 4.18      # J/K
T0 = 5        # initial temperature

# Define the ODE
def dTdt(t, T):
    T_out = np.sin(t / 10) * 10 + 20  # outside temperature varies sinusoidally
    return (T_out - T) / (R * C)

t_span = (0, 100)  # time span for the simulation

# Solve the ODE
sol = solve_ivp(dTdt, t_span, [T0], t_eval=np.linspace(t_span[0], t_span[1], 500))

# Plot results
plt.plot(sol.t, sol.y[0], label="Room temperature")
plt.plot(sol.t, np.sin(sol.t / 10) * 10 + 20, label="Outside temperature")
plt.xlabel("Time [s]")
plt.ylabel("Temperature [°C]")
plt.title("Room Temperature vs Time")
plt.legend()
plt.grid(True)
plt.show()