import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import pandas as pd
import matplotlib.pyplot as plt

# Load thermal resistance data from file
r_val = np.loadtxt('thermal_resistance/out/R_results_Expanded Polystyrene (EPS)_mineral_wool_Polyurethane (PUR).txt', usecols=(3,), skiprows=1)
print (r_val)

# Load Fourier series equation for ambient temperature from file
with open('MIDAS/out/symbolic_fourier_temperature.txt', 'r') as f:
        fourier_equation_string = f.read().strip()
print (fourier_equation_string)
corrected_equation_string = fourier_equation_string.replace('sin', 'np.sin')
corrected_equation_string = corrected_equation_string.replace('cos', 'np.cos')
corrected_equation_string = corrected_equation_string.replace('pi', 'np.pi')

# Load solar irradiance data from file
try:
    df_q_val = pd.read_excel('SolRad.xlsx', skiprows=2, usecols=[4], header=None)
    q_val = df_q_val.iloc[:, 0].values
    if len(q_val) == 17520: #to avoid the random error where data ended up being half hourly????
        q_val = q_val.reshape(-1, 2).mean(axis=1)
        print(f"Downsampled q_val to hourly ({len(q_val)} points)")
    q_val = q_val.flatten()
    q_val = q_val[~np.isnan(q_val)]  # remove NaNs
    print("Final q_val shape:", q_val.shape)

except OSError:
    print ("File not found. Please ensure the solar irradiance data file exists at the specified path.")
    df_q_val = pd.DataFrame()  # Assign an empty DataFrame if file not found

# Parameters
width_house = 9.0  # meters
length_house = 4.0 # meters
height_house = 3.0 # meters
thickness_wall = 0.1 # meters

area_wall = 2 * height_house * (length_house + width_house)  # m2

# Thermal properties
h_o = 35 # outside convective heat transfer coefficient W/m2K
h_i = 8 # inside convective heat transfer coefficient W/m2K

R_conv_outside = 1 / (h_o * area_wall) # outside convective resistance
R_conv_inside = 1 / (h_i * area_wall) # inside convective resistance
R_eff = R_conv_outside + R_conv_inside + r_val 

n_hours = len(q_val)
time_hours = np.arange(n_hours)          # time in hours
time_seconds = time_hours * 3600         # convert to seconds for Fourier-based functions

def T_amb(t_hours):
    return eval(
        corrected_equation_string, 
        {'np': np}, 
        {'t': t_hours}
        )


# Evaluate hourly
T_amb_hourly = T_amb(time_hours)
T_amb_hourly = np.asarray(T_amb_hourly).flatten() #ensure 1D array to avoid shape issues

# Calculations
def calculate_T_out_eff(r_val, q_val, T_amb_hourly):

    #for debugging
    print(">>> Entering calculate_T_out_eff")
    print("  q_val:", len(q_val))
    print("  T_amb_hourly:", len(T_amb_hourly))
    ##

    n_hours = len(q_val)
    n_mat = len(r_val)
    T_out_eff = np.zeros((n_hours, n_mat))
    R_eff_all = np.zeros(n_mat)

    for j, R_wall in enumerate(r_val):
        R_eff_all[j] = (R_conv_outside + R_conv_inside + R_wall)/(12)
        T_out_eff[:, j] = T_amb_hourly + q_val * R_conv_outside
    
    print("  T_out_eff shape:", T_out_eff.shape)
    return T_out_eff, R_eff_all

T_out_eff, R_eff = calculate_T_out_eff(r_val, q_val, T_amb_hourly)


time_index = pd.RangeIndex(start=0, stop=n_hours, step=1, name="hour")
columns = [f"R={R:.2f}" for R in r_val]

T_out_df = pd.DataFrame(T_out_eff, index=time_index, columns=columns)
T_out_df.attrs['R_eff'] = dict(zip(columns, R_eff))

print("T_out_eff shape:", T_out_eff.shape)
print("T_out_df shape:", T_out_df.shape)
print("First few rows:\n", T_out_df.head())
print("Index length:", len(T_out_df.index))
print("Columns:", len(T_out_df.columns))

output_path = "T_out_effective_temperatures.xlsx"

with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
    # Sheet 1: Hourly effective wall temperatures for each material
    T_out_df.to_excel(writer, sheet_name="T_out_eff_hourly")

    # Sheet 2: Effective resistances (metadata)
    R_eff_dict = T_out_df.attrs['R_eff']
    R_eff_df = pd.DataFrame(list(R_eff_dict.items()), columns=["Material (R_label)", "R_eff_total"])
    R_eff_df.to_excel(writer, sheet_name="R_eff_summary", index=False)
