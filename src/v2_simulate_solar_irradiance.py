import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import pandas as pd
import matplotlib.pyplot as plt

# Load R-values for wall from file
try:
    r_val = np.loadtxt('thermal_resistance/out/R_results_Expanded Polystyrene (EPS)_mineral_wool_Polyurethane (PUR).txt', usecols=(3,), skiprows=1)
except OSError:
    print ("File not found. Please ensure the R-values file exists at the specified path.")
    r_val = np.array([])  # Assign an empty array if file not found

# Load solar irradiance data from file
try:
    q_val = np.loadtxt('solar_irradiance/out/solar_irradiance_data.txt', usecols=(1,), skiprows=1)
except OSError:
    print ("File not found. Please ensure the solar irradiance data file exists at the specified path.")
    r_val = np.array([])  # Assign an empty array if file not found

# Parameters
width_house = 9.0  # meters
length_house = 4.0 # meters
height_house = 3.0 # meters
thickness_wall = 0.1 # meters

area_wall = 2 * height_house * (length_house + width_house)  # m2

# Thermal properties
h_o = 35 # outside convective heat transfer coefficient W/m2K
h_i = 8 # inside convective heat transfer coefficient W/m2K
R_wall = 5.3 # PLACEHOLDER for testing code. Replace with i in r_val.

R_conv_outside = 1 / (h_o * area_wall) # outside convective resistance
R_conv_inside = 1 / (h_i * area_wall) # inside convective resistance
R_eff = R_conv_outside + R_conv_inside + R_wall 

n_hours = len(q_val)
time_hours = np.arange(n_hours)          # time in hours
time_seconds = time_hours * 3600         # convert to seconds for Fourier-based functions

#Insert Fourier series of T_amb here

T_amb_hourly = T_amb(time_seconds)

# Calculations
def calculate_T_out_eff(t):  #Returns a 2D array [n_hours x n_materials] of effective wall temperatures.
    n_hours = len(q_val)
    n_mat = len(r_val)
    T_out_eff = np.zeros((n_hours, n_mat))
    for j, R_wall in enumerate(r_val):
        R_eff = R_conv_outside + R_conv_inside + R_wall
        T_out_eff[:, j] = T_amb_hourly + q_val * R_conv_outside
    return T_out_eff, R_eff

T_out_eff = calculate_T_out_eff(r_val, q_val, T_amb_hourly)
time_index = pd.RangeIndex(start=0, stop=n_hours, step=1, name="hour")
T_out_df = pd.DataFrame(T_out_eff, index=time_index, columns=[f"R={R:.2f}" for R in r_val])

