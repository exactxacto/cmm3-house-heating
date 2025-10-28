import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Load solar irradiance data from file
q_val = range(0, 500, 50)  # PLACEHOLDER for solar irradiance values in Watts/m2

# Parameters
width_house = 9.0  # meters
length_house = 4.0 # meters
height_house = 3.0 # meters
thickness_wall = 0.1 # meters

area_wall = 2 * height_house * (length_house + width_house)  # m2

# Thermal properties
h_o = 35 # outside convective heat transfer coefficient W/m2K
h_i = 8 # inside convective heat transfer coefficient W/m2K
R_wall = 5.3 # PLACEHOLDER for testing code

R_conv_outside = 1 / (h_o * area_wall) # outside convective resistance
R_conv_inside = 1 / (h_i * area_wall) # inside convective resistance
R_eff = R_conv_outside + R_conv_inside + R_wall 


# Calculations
def outside_temp(t):
    T_amb = np.sin(t / (3600 * 12)) * 10 + 20  # one cycle every 24 hours
    for i in q_val:
        q = i
        T_out_eff = T_amb + q * R_conv_outside
    return T_out_eff

for t in range(0, 86400, 3600):
    print(f"At time {t} seconds, effective outside temperature is {outside_temp(t)} °C")

plt.plot(range(0, 86400, 3600), [outside_temp(t) for t in range(0, 86400, 3600)])
plt.xlabel('Time (seconds)')
plt.ylabel('Effective Outside Temperature (°C)')

plt.show()