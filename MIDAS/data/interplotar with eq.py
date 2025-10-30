#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 29 12:17:00 2025

@author: wanhaziq
"""

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from datetime import datetime
import sys # Added for sys.exit()

# --- 1. Data Preparation and Loading ---

def load_data():
    """
    ATTEMPTS to load your actual hourly temperature data from 'EdiTempYear.xlsx'.
    
    If the file is not found or the columns are incorrect, the script will 
    print an error and halt execution. IT WILL NOT FALL BACK TO DUMMY DATA.
    
    NOTE: You must ensure 'openpyxl' is installed (pip install openpyxl) 
    and the Excel file is in the same directory as this script.
    """
    
    # Custom file path and column names based on user request
    file_path = 'EdiTempYear.xlsx'
    time_column = 'ob_time'
    temp_column = 'air_temperature'
    
    try:
        print(f"Attempting to load data from: {file_path}...")
        
        # 1. Read the Excel file.
        df = pd.read_excel(file_path, sheet_name=0) 
        
        # 2. Rename columns to standardized names for the rest of the script
        df = df.rename(columns={
            time_column: 'Time', 
            temp_column: 'Temperature'
        })
        
        # Check if the required columns exist after renaming
        if 'Time' not in df.columns or 'Temperature' not in df.columns:
            raise KeyError(f"Required columns ('{time_column}', '{temp_column}') not found or incorrectly named.")
            
        # 3. Set the date/time column as the index.
        df.set_index('Time', inplace=True)

        # 4. Resample to hourly data (crucial for the 8760 hour model)
        # This takes the mean if multiple readings are in one hour.
        df = df['Temperature'].resample('H').mean().to_frame() 
        # Keeping this internal dropna for initial cleaning
        df.dropna(subset=['Temperature'], inplace=True) 

        # Final size check (2023 has 8760 hours)
        print(f"Data successfully loaded for {len(df)} hours.")
        if len(df) < 8700:
            print("WARNING: Significant data gaps detected (less than 8700 hours).")

        return df
        
    except FileNotFoundError:
        print(f"\n--- FATAL ERROR: FILE NOT FOUND ---")
        print(f"Could not find the Excel file: '{file_path}'")
        print("ACTION: Ensure the file is in the same folder as this Python script.")
        sys.exit(1)
    except KeyError as e:
        print(f"\n--- FATAL ERROR: COLUMN NAMES ---")
        print(f"A column error occurred: {e}")
        print(f"ACTION: Check that your spreadsheet headers are exactly '{time_column}' and '{temp_column}'.")
        sys.exit(1)
    except Exception as e:
        print(f"\n--- FATAL ERROR: GENERIC ERROR ---")
        print(f"An unexpected error occurred during data loading: {type(e).__name__}: {e}")
        print("ACTION: Check data integrity or ensure 'openpyxl' is installed (pip install openpyxl).")
        sys.exit(1)

# Load the data. If this fails, the script will exit here.
data_df = load_data() 

# --- CRITICAL FIX: Robust Cleaning for NaN/Inf values ---
initial_len = len(data_df)
# Replace infinity values with NaN
data_df.replace([np.inf, -np.inf], np.nan, inplace=True)
# Drop any row that now contains NaN (including ones converted from inf)
data_df.dropna(subset=['Temperature'], inplace=True)
cleaned_len = len(data_df)

if initial_len != cleaned_len:
    print(f"\nALERT: Dropped {initial_len - cleaned_len} hours due to NaN or Inf values after final cleaning.")
# ---------------------------------------------------

# The time_hours array must start from 0 and increment by 1 for each data point
# NOTE: time_hours array must be created *after* dropping bad rows
time_hours = np.arange(len(data_df)) 
temp_data = data_df['Temperature'].values


# --- 2. Defining the Interpolation Model ---

def two_scale_sine_model(t, C0, C_annual, phi_annual, C_diurnal, phi_diurnal):
    """
    A composite sine wave model to fit both annual (8760h period) and 
    diurnal (24h period) variations simultaneously.
    """
    
    # Annual component (8760 hours in 2023)
    annual_component = C_annual * np.sin(2 * np.pi * t / 8760 + phi_annual)
    
    # Diurnal component (24 hours)
    diurnal_component = C_diurnal * np.sin(2 * np.pi * t / 24 + phi_diurnal)
    
    return C0 + annual_component + diurnal_component


# --- 3. Curve Fitting Execution ---

# Initial parameter guesses (p0) are based on a typical temperate climate.
# [C0 (Mean), C_annual (Annual Amplitude), phi_annual (Annual Phase), 
# C_diurnal (Daily Amplitude), phi_diurnal (Daily Phase)]
initial_guess = [10.0, 6.0, 0.0, 3.0, -1.0] 

try:
    print("\nBeginning curve fitting process...")
    popt, pcov = curve_fit(
        two_scale_sine_model, 
        time_hours, 
        temp_data, 
        p0=initial_guess,
        maxfev=5000 
    )
    
    # Extract fitted parameters
    C0_fit, C_annual_fit, phi_annual_fit, C_diurnal_fit, phi_diurnal_fit = popt

    print("\n--- Fitted Parameters (Mathematical Model) ---")
    print(f"Annual Mean (C0): {C0_fit:.4f} °C")
    print(f"Annual Amplitude (C_annual): {C_annual_fit:.4f} °C")
    print(f"Annual Phase (phi_annual): {phi_annual_fit:.4f} rad")
    print(f"Diurnal Amplitude (C_diurnal): {C_diurnal_fit:.4f} °C")
    print(f"Diurnal Phase (phi_diurnal): {phi_diurnal_fit:.4f} rad")
    
    # --- NEW: Print the Final Equation ---
    equation_string = (
        f"\n--- Final Interpolation Equation (T(t) in °C) ---\n"
        f"T(t) = {C0_fit:.4f} "
        f"+ {C_annual_fit:.4f} * sin( (2\u03c0 t / 8760) + {phi_annual_fit:.4f} ) "
        f"+ {C_diurnal_fit:.4f} * sin( (2\u03c0 t / 24) + {phi_diurnal_fit:.4f} )"
    )
    print(equation_string)
    # -------------------------------------

    # Generate the fitted curve data
    temp_fit = two_scale_sine_model(time_hours, *popt)
    print("\nFitting successful. Generating plots.")

except RuntimeError as e:
    print(f"\n--- FATAL ERROR: FITTING FAILED ---")
    print(f"Curve fitting failed. This often happens with very noisy or bad data.")
    print(f"Error detail: {e}")
    sys.exit(1)


# --- 4. Plotting the Results ---

plt.style.use('seaborn-v0_8-darkgrid')
plt.figure(figsize=(15, 8))

# A. Plotting the full year
plt.subplot(2, 1, 1)
plt.plot(data_df.index, temp_data, label='Raw Hourly Data', color='gray', alpha=0.5, linewidth=0.5)
plt.plot(data_df.index, temp_fit, label='Fitted Interpolation Curve', color='#007BFF', linewidth=2)

# Calculate and plot the long-term trend (diurnal component removed)
long_term_trend = two_scale_sine_model(time_hours, C0_fit, C_annual_fit, phi_annual_fit, 0, 0)
plt.plot(data_df.index, long_term_trend, label='Long-Term Seasonal Trend', color='#FF5733', linestyle='--', linewidth=1.5)

plt.title('2023 Edinburgh Temperature: Full Year Interpolation', fontsize=16)
plt.xlabel('Date (2023)', fontsize=12)
plt.ylabel('Temperature (°C)', fontsize=12)
plt.legend(loc='upper right')

# B. Zooming in on a single week
# You can change the starting day here to examine different parts of the year
start_day_zoom = 200 # Day 200 is roughly mid-July
start_hour_zoom = start_day_zoom * 24
end_hour_zoom = start_hour_zoom + 24 * 7 # 7 days
zoom_slice = slice(start_hour_zoom, end_hour_zoom)

plt.subplot(2, 1, 2)
plt.plot(data_df.index[zoom_slice], temp_data[zoom_slice], label='Raw Data (Zoom)', color='gray', alpha=0.7, marker='.', markersize=5)
plt.plot(data_df.index[zoom_slice], temp_fit[zoom_slice], label='Fitted Curve (Zoom)', color='#007BFF', linewidth=2.5)
plt.title(f'Zoomed View: Week Starting Day {start_day_zoom} (Mid-Summer Diurnal Cycle)', fontsize=14)
plt.xlabel('Date and Time', fontsize=12)
plt.ylabel('Temperature (°C)', fontsize=12)
plt.legend(loc='upper right')

plt.tight_layout()
plt.show()

print("\n--- Script Execution Complete ---")
