import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from datetime import datetime

# --- 1. Data Preparation and Loading ---

def load_data():
    """
    Loads your actual hourly temperature data for the year 2023.
    
    IMPORTANT: You must UNCOMMENT the 'Excel Loading Block' below and 
    update the filename and column names to match your Excel file.
    
    Ensure your data is a single time series of hourly temperatures in Celsius.
    If you receive a 'ModuleNotFoundError' when loading Excel, you may need 
    to install the 'openpyxl' library (pip install openpyxl).
    """
    
    # =========================================================================
    #            EXCEL LOADING BLOCK (Uncomment and Customize this)
    # =========================================================================
    """
    try:
        # 1. Update your Excel file path here:
        file_path = 'your_2023_temperature_data.xlsx'
        
        # 2. Read the Excel file. Adjust 'sheet_name' if necessary.
        df = pd.read_excel(file_path, sheet_name=0) 
        
        # 3. Rename columns to match the script's expectations:
        # Assuming one column is 'Time' (or 'Date/Time') and another is 'Temperature'
        # Set the date/time column as the index.
        df = df.rename(columns={
            'Your_DateTime_Column_Name': 'Time', 
            'Your_Temperature_Column_Name': 'Temperature'
        })
        df.set_index('Time', inplace=True)

        # 4. Ensure the data is sorted by time and resampled to hourly if needed (optional)
        # The model assumes 8760 points (hourly for 2023).
        df = df['Temperature'].resample('H').mean().to_frame() 

        # Check for missing hours (should be 8760 for a full year)
        if len(df) != 8760:
            print(f"WARNING: Data has {len(df)} hours. Expected 8760 for 2023.")
            # Drop NaN values if resampling introduced them
            df.dropna(subset=['Temperature'], inplace=True)
            
        print(f"Data loaded successfully from {file_path} for {len(df)} hours.")
        return df
        
    except FileNotFoundError:
        print(f"ERROR: Excel file '{file_path}' not found. Switching to dummy data.")
    except Exception as e:
        print(f"ERROR loading Excel data: {e}. Switching to dummy data.")
    """
    # =========================================================================
    
    # --- DUMMY DATA GENERATION (Used if Excel loading is commented out or fails) ---
    print("Generating synthetic hourly data for 2023...")
    
    # 2023 has 8760 hours (not a leap year)
    total_hours = 8760
    time_hours = np.arange(total_hours)

    C0 = 10.0      # Annual Mean Temperature (°C)
    C_annual = 6.0 # Amplitude of the annual swing (°C)
    C_diurnal = 3.0 # Amplitude of the daily swing (°C)
    
    # 1. Seasonal Component (Period = 8760 hours)
    seasonal_phase_shift = 4800 
    seasonal_component = C_annual * np.sin(2 * np.pi * (time_hours - seasonal_phase_shift) / 8760)

    # 2. Diurnal Component (Period = 24 hours)
    diurnal_phase_shift = 15
    diurnal_component = C_diurnal * np.sin(2 * np.pi * (time_hours - diurnal_phase_shift) / 24)
    
    modulation_factor = 0.5 * seasonal_component / C_annual + 0.5
    
    # 3. Combine components and add noise
    temperatures = (
        C0 
        + seasonal_component 
        + (diurnal_component * modulation_factor) 
        + np.random.normal(0, 1.5, total_hours)
    )
    
    start_date = datetime(2023, 1, 1, 0)
    time_index = pd.date_range(start=start_date, periods=total_hours, freq='H')
    
    df = pd.DataFrame({'Temperature': temperatures}, index=time_index)
    
    print(f"Data generated for {len(df)} hours.")
    return df

# Load the data - Call the function to load either Excel or dummy data
data_df = load_data()
# The time_hours array must start from 0 and increment by 1 for each data point
time_hours = np.arange(len(data_df)) 
temp_data = data_df['Temperature'].values


# --- 2. Defining the Interpolation Model ---

def two_scale_sine_model(t, C0, C_annual, phi_annual, C_diurnal, phi_diurnal):
    """
    A composite sine wave model to fit both annual and diurnal variations.
    
    T(t) = C0 + C_annual * sin(2*pi*t / 8760 + phi_annual) 
               + C_diurnal * sin(2*pi*t / 24 + phi_diurnal)
               
    The annual period is fixed at 8760 hours (for 2023).
    The diurnal period is fixed at 24 hours.
    """
    
    # Annual component (8760 hours in 2023)
    annual_component = C_annual * np.sin(2 * np.pi * t / 8760 + phi_annual)
    
    # Diurnal component (24 hours)
    diurnal_component = C_diurnal * np.sin(2 * np.pi * t / 24 + phi_diurnal)
    
    return C0 + annual_component + diurnal_component


# --- 3. Curve Fitting Execution ---

# Initial parameter guesses (p0) are crucial for curve_fit to succeed.
# [C0, C_annual, phi_annual, C_diurnal, phi_diurnal]
initial_guess = [10.0, 6.0, 0.0, 3.0, -1.0] 

try:
    # Perform the least-squares fit
    popt, pcov = curve_fit(
        two_scale_sine_model, 
        time_hours, 
        temp_data, 
        p0=initial_guess,
        maxfev=5000 
    )
    
    # Extract fitted parameters
    C0_fit, C_annual_fit, phi_annual_fit, C_diurnal_fit, phi_diurnal_fit = popt

    print("\n--- Fitted Parameters ---")
    print(f"Annual Mean (C0): {C0_fit:.2f} °C")
    print(f"Annual Amplitude (C_annual): {C_annual_fit:.2f} °C")
    print(f"Annual Phase (phi_annual): {phi_annual_fit:.2f} rad")
    print(f"Diurnal Amplitude (C_diurnal): {C_diurnal_fit:.2f} °C")
    print(f"Diurnal Phase (phi_diurnal): {phi_diurnal_fit:.2f} rad")

    # Generate the fitted curve data
    temp_fit = two_scale_sine_model(time_hours, *popt)

except RuntimeError as e:
    print(f"\nERROR: Curve fitting failed. Check initial guess or data quality.")
    print(e)
    temp_fit = np.zeros_like(temp_data) 


# --- 4. Plotting the Results ---

plt.style.use('seaborn-v0_8-darkgrid')
plt.figure(figsize=(15, 8))

# A. Plotting the full year
plt.subplot(2, 1, 1)
plt.plot(data_df.index, temp_data, label='Raw Hourly Data', color='gray', alpha=0.5, linewidth=0.5)
plt.plot(data_df.index, temp_fit, label='Fitted Curve ($T(t)$)', color='#FF5733', linewidth=2)

# Calculate and plot the long-term trend (diurnal component removed)
# Note: Using 0 for diurnal parameters isolates the annual trend.
long_term_trend = two_scale_sine_model(time_hours, C0_fit, C_annual_fit, phi_annual_fit, 0, 0)
plt.plot(data_df.index, long_term_trend, label='Long-Term Seasonal Trend ($\mu(t)$)', color='#3366FF', linestyle='--', linewidth=1.5)

plt.title('2023 Temperature: Long-Term Trend and Diurnal Variation', fontsize=16)
plt.xlabel('Date (2023)', fontsize=12)
plt.ylabel('Temperature (°C)', fontsize=12)
plt.legend(loc='upper right')

# B. Zooming in on a single week (to clearly show the daily variation)
# We choose a spot in July for better visibility in the dummy data
start_day_zoom = 200 
start_hour_zoom = start_day_zoom * 24
end_hour_zoom = start_hour_zoom + 24 * 7 # 7 days
zoom_slice = slice(start_hour_zoom, end_hour_zoom)

plt.subplot(2, 1, 2)
plt.plot(data_df.index[zoom_slice], temp_data[zoom_slice], label='Raw Data (Zoom)', color='gray', alpha=0.7, marker='.', markersize=5)
plt.plot(data_df.index[zoom_slice], temp_fit[zoom_slice], label='Fitted Curve (Zoom)', color='#FF5733', linewidth=2.5)
plt.title(f'Zoomed View: Week of Day {start_day_zoom} Showing Day/Night Variation', fontsize=14)
plt.xlabel('Date and Time', fontsize=12)
plt.ylabel('Temperature (°C)', fontsize=12)
plt.legend(loc='upper right')

plt.tight_layout()
plt.show()

# --- 5. Exporting the Interpolated Data (Optional) ---
interpolated_df = pd.DataFrame({
    'Time': data_df.index,
    'Fitted_Temperature': temp_fit
})
# interpolated_df.to_csv('interpolated_2023_temp_fit.csv', index=False)

print("\n--- Script Execution Complete ---")
