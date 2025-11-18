import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import pandas as pd

from v2_simulate_solar_irradiance import T_out_df, R_eff_dict, time_hours, n_hours

try:
    T_out_df  # defined in v2_simulate_solar_irradiance
except NameError:
    raise RuntimeError("T_out_df not found — run the generation code first.")

columns = list(T_out_df.columns)
n_hours = len(T_out_df.index)
time_hours = np.asarray(T_out_df.index)           # 0..n_hours-1

# Extract R_eff mapping (dictionary keyed by column names)
R_eff_dict = T_out_df.attrs.get('R_eff', None)
if R_eff_dict is None:
    try:
        R_eff  # numpy array from previous code
        R_eff_dict = dict(zip(columns, R_eff))
    except NameError:
        raise RuntimeError("R_eff metadata not found. Put R_eff into T_out_df.attrs['R_eff'] or define R_eff array.")

# Parameters
width_house = 9.0  # meters
length_house = 4.0 # meters
height_house = 3.0 # meters

# Calculations
vol_house = width_house * length_house * height_house  # m^3
density_air = 1.225  # kg/m^3
mass_air = vol_house * density_air  # kg
C = mass_air * 1005  + 1500000  # J/K (including thermal mass of )
T0 = 25   # initial indoor temperature [°C]

print (C)

# Assuming n_hours is 8760 (set correctly at the top of solve_ODE)

def simulate(R_label):
    # 1. Extract data series
    T_out_series = np.ravel(T_out_df[R_label].to_numpy().astype(float))
    R_eff = float(R_eff_dict[R_label])
    
    # 2. Get the expected lengths
    expected_length = len(time_hours) # This should be 8760
    current_length = len(T_out_series)

    # 3. Error handling - downsample due to weird length issues with some data
    if current_length != expected_length:
        if current_length % expected_length == 0 and expected_length > 0:
            # Determine the factor (2x, 3x, etc.)
            factor = current_length // expected_length
            
            print(f"**WARNING: Downsampling {current_length} points for {R_label} (Factor {factor}x) to {expected_length}.**")
            
            # Reshape and average: Group points by the factor and take the mean
            T_out_series = T_out_series.reshape(-1, factor).mean(axis=1)
            
        else:
            # If the length is not a clean multiple, something else is messed up.
            raise RuntimeError(f"Length mismatch persists for {R_label}: Index={expected_length}, Data={current_length}. Data length is not a clean multiple of the index length.")
    
    # Check if lengths are correct now
    print(f"{R_label}: len(time_hours)={len(time_hours)}, len(T_out_series)={len(T_out_series)}")

    def dTdt(t, T):
        hour = t / 3600
        T_out_t = np.interp(hour, time_hours, T_out_series)
        return (T_out_t - T) / (R_eff * C)

    # One-year simulation (8760 hours)
    t_span = (0, expected_length * 3600)
    t_eval = np.arange(0, expected_length * 3600, 3600)
    sol = solve_ivp(dTdt, t_span, [T0], t_eval=t_eval, method='RK45')

    T_hourly = sol.y[0]
    t_hourly = sol.t / 3600
    return t_hourly, T_hourly

# Run simulations for all materials 
output_rows = []
for R_label in columns:
    t_hourly, T_hourly = simulate(R_label)
    R_val = R_eff_dict[R_label]
    row_data = np.concatenate(([R_val], T_hourly))
    output_rows.append(row_data)

results_array = np.vstack(output_rows)  # shape = (n_materials, n_hours + 1)

time_columns = [f'Hour {int(t)}' for t in t_hourly]
all_columns = ['R_eff_total'] + time_columns
results_df = pd.DataFrame(results_array, columns=all_columns)
output_file_name = 'simulation_results_T_indoor.xlsx' 

results_df.to_excel(output_file_name, index=False, sheet_name='Indoor_Temp_Simulation')
print ("Simulation complete. Hopefully no errors this time!!!!")

# -------------------------------------------------------------
# PLOTTING FROM results_df (no re-simulation)
# -------------------------------------------------------------

# Sort labels by R-value
sorted_labels = sorted(columns, key=lambda x: R_eff_dict[x])
low_label  = sorted_labels[0]
mid_label  = sorted_labels[len(sorted_labels)//2]
high_label = sorted_labels[-1]

# Map labels to row indices in results_df
label_to_row = {label: i for i, label in enumerate(columns)}

# Extract indoor temperature series directly from results_df
low_row  = results_df.iloc[label_to_row[low_label]].to_numpy()
mid_row  = results_df.iloc[label_to_row[mid_label]].to_numpy()
high_row = results_df.iloc[label_to_row[high_label]].to_numpy()

# First element is R_eff_total; the rest are hourly temps
T_low  = low_row[1:]
T_mid  = mid_row[1:]
T_high = high_row[1:]

# Time vector
hours = np.arange(len(T_low))

# Outdoor temp (choose any column; the outdoor index is consistent)
T_outdoor = T_out_df[low_label].to_numpy().astype(float)

plt.figure(figsize=(12, 6))

# Outdoor in bold red
plt.plot(
    hours, T_outdoor,
    color='red', linewidth=2.2,
    label='Outdoor Temperature'
)

# Indoor temperatures in three blue shades
plt.plot(
    hours, T_low,
    color='#9ecae1', linewidth=1.4,
    label=f'Indoor (Low R: {R_eff_dict[low_label]:.3f})'
)
plt.plot(
    hours, T_mid,
    color='#4292c6', linewidth=1.4,
    label=f'Indoor (Mid R: {R_eff_dict[mid_label]:.3f})'
)
plt.plot(
    hours, T_high,
    color='#084594', linewidth=1.4,
    label=f'Indoor (High R: {R_eff_dict[high_label]:.3f})'
)

plt.xlabel("Time [hours]")
plt.ylabel("Temperature [°C]")
plt.title("Indoor vs Outdoor Temperature Over One Year")
plt.grid(True, alpha=0.25)
plt.legend()
plt.tight_layout()
plt.show()
