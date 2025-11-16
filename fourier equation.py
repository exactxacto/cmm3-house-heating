#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 29 12:46:00 2025

@author: wanhaziq
"""

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from datetime import datetime
import sys 

# --- 1. Data Preparation and Loading ---

def load_data():
    """
    ATTEMPTS to load actual hourly temperature data from 'EdiTempYear.xlsx'.
    
    If the file is not found or the columns are incorrect, the script will 
    print an error and halt execution.
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
        df = df['Temperature'].resample('H').mean().to_frame() 
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
data_df.replace([np.inf, -np.inf], np.nan, inplace=True)
data_df.dropna(subset=['Temperature'], inplace=True)
cleaned_len = len(data_df)

if initial_len != cleaned_len:
    print(f"\nALERT: Dropped {initial_len - cleaned_len} hours due to NaN or Inf values after final cleaning.")
# ---------------------------------------------------

time_hours = np.arange(len(data_df)) 
temp_data = data_df['Temperature'].values


# --- 2. Defining the NEW Interpolation Model (11 Parameters - Added 8h Harmonic) ---

def two_scale_sine_model(t, C0, C_annual, phi_annual, A_base, A_seasonal, phi_amp, phi_diurnal, A_harm, phi_harm, A_harm3, phi_harm3):
    """
    11-parameter model: Combines annual trend, a seasonally-modulated diurnal 
    (24h) cycle, a 12-hour harmonic component, AND an 8-hour harmonic component
    for the best possible fit fidelity.
    """
    
    # 1. Long-Term Seasonal Trend 
    mu_t = C0 + C_annual * np.sin(2 * np.pi * t / 8760 + phi_annual)
    
    # 2. Seasonally Varying Diurnal Amplitude (24-hour cycle)
    diurnal_amplitude = A_base + A_seasonal * np.sin(2 * np.pi * t / 8760 + phi_amp)
    diurnal_component = diurnal_amplitude * np.sin(2 * np.pi * t / 24 + phi_diurnal)
    
    # 3. Second Diurnal Harmonic (12-hour cycle)
    harmonic_component = A_harm * np.sin(2 * np.pi * t / 12 + phi_harm)
    
    # 4. Third Diurnal Harmonic (8-hour cycle) - NEW TERM
    harmonic_component_3 = A_harm3 * np.sin(2 * np.pi * t / 8 + phi_harm3)
    
    return mu_t + diurnal_component + harmonic_component + harmonic_component_3


# --- 3. Curve Fitting Execution ---

# Updated Initial Guesses (11 parameters now)
# [..., A_harm3 (NEW), phi_harm3 (NEW)]
initial_guess = [10.0, 6.0, 0.0, 2.0, 1.0, 0.0, -1.0, 0.5, 0.0, 0.2, 0.0] 

try:
    print("\nBeginning curve fitting process with 11-parameter model (including 8h harmonic)...")
    popt, pcov = curve_fit(
        two_scale_sine_model, 
        time_hours, 
        temp_data, 
        p0=initial_guess,
        maxfev=20000 # Increased iterations for the most complex model
    )
    
    # Extract fitted parameters (11 variables now)
    C0_fit, C_annual_fit, phi_annual_fit, A_base_fit, A_seasonal_fit, phi_amp_fit, phi_diurnal_fit, A_harm_fit, phi_harm_fit, A_harm3_fit, phi_harm3_fit = popt

    print("\n--- Fitted Parameters (Advanced Mathematical Model with Harmonics) ---")
    print(f"1. Annual Mean (C0): {C0_fit:.4f} °C")
    print(f"2. Annual Temp Amplitude (C_annual): {C_annual_fit:.4f} °C")
    print(f"3. Annual Temp Phase (phi_annual): {phi_annual_fit:.4f} rad")
    print("-" * 40)
    print(f"4. Base Diurnal Amplitude (A_base): {A_base_fit:.4f} °C")
    print(f"5. Seasonal Diurnal Modulation (A_seasonal): {A_seasonal_fit:.4f} °C")
    print(f"6. Diurnal Amplitude Phase (phi_amp): {phi_amp_fit:.4f} rad")
    print(f"7. Diurnal Time Phase (phi_diurnal): {phi_diurnal_fit:.4f} rad")
    print("-" * 40)
    print(f"8. 12-hour Harmonic Amplitude (A_harm): {A_harm_fit:.4f} °C")
    print(f"9. 12-hour Harmonic Phase (phi_harm): {phi_harm_fit:.4f} rad")
    print("-" * 40)
    print(f"10. 8-hour Harmonic Amplitude (A_harm3): {A_harm3_fit:.4f} °C")
    print(f"11. 8-hour Harmonic Phase (phi_harm3): {phi_harm3_fit:.4f} rad")
    
    # --- Print the Final Equation ---
    # Simplified string formatting for console readability
    term_A = f"({C0_fit:.4f} + {C_annual_fit:.4f}*sin( (2\u03c0 t / 8760) + {phi_annual_fit:.4f} ))"
    term_B_amp = f"({A_base_fit:.4f} + {A_seasonal_fit:.4f}*sin( (2\u03c0 t / 8760) + {phi_amp_fit:.4f} ))"
    term_C_diurnal = f"sin( (2\u03c0 t / 24) + {phi_diurnal_fit:.4f} )"
    term_D_harm = f"{A_harm_fit:.4f} * sin( (2\u03c0 t / 12) + {phi_harm_fit:.4f} )"
    term_E_harm3 = f"{A_harm3_fit:.4f} * sin( (2\u03c0 t / 8) + {phi_harm3_fit:.4f} )" # NEW TERM
    
    equation_string = (
        f"\n--- Final Interpolation Equation (T(t) in °C) ---\n"
        f"T(t) = Annual_Trend(t) + Diurnal_Fundamental(t) + Diurnal_Harmonic_12h(t) + Diurnal_Harmonic_8h(t)\n"
        f"T(t) = {term_A}\n"
        f"     + {term_B_amp} * {term_C_diurnal}\n"
        f"     + {term_D_harm}\n"
        f"     + {term_E_harm3}" # Added new harmonic term
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

# Calculate and plot the long-term trend (The seasonal part)
# This excludes the entire diurnal component
long_term_trend = C0_fit + C_annual_fit * np.sin(2 * np.pi * time_hours / 8760 + phi_annual_fit)
plt.plot(data_df.index, long_term_trend, label='Long-Term Seasonal Trend', color='#FF5733', linestyle='--', linewidth=1.5)

plt.title('2023 Edinburgh Temperature: Full Year Interpolation (11-Parameter Harmonic Model)', fontsize=16)
plt.xlabel('Date (2023)', fontsize=12)
plt.ylabel('Temperature (°C)', fontsize=12)
plt.legend(loc='upper right')

# B. Zooming in on a single week
start_day_zoom = 200 # Day 200 is roughly mid-July
start_hour_zoom = start_day_zoom * 24
end_hour_zoom = start_hour_zoom + 24 * 7 
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
