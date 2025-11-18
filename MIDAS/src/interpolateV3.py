# -*- coding: utf-8 -*-
"""
Created on Tue Nov 18 12:13:58 2025

@author: yaya0
"""

from pathlib import Path
import numpy as np
import sympy as sp
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------------
#Load Data
# ----------------------------------

#Find the repository root
def repo_root() -> Path:
    if "__file__" in globals():
        return Path(__file__).resolve().parent.parent
    else:
        return Path.cwd()
    


#Loads Excel data from the 'data' folder and specified sheet 'Raw_Data'
def load_data(filename="EdiTempYear.xlsx", sheet_name="Raw_Data"):
    base = repo_root()
    data_path = base / "data" / filename

    if not data_path.exists():
        raise FileNotFoundError(f"Could not find {data_path}")

    df = pd.read_excel(data_path, sheet_name=sheet_name)
    return df

# ----------------------------------
#Clean Temperature Values
# ----------------------------------
#Extract temperature from EdiTemp Data Sheet
#Converts to date-time column using Panads
#Interpolate any missing values
def extract_temperature(df, date_col="ob_time", temp_col="air_temperature",
                        interp_limit=6):
    
    #Look for date column
    if date_col not in df.columns:
        raise KeyError(f"Date column {date_col} not found in DataFrame")


    #Extract date and temperature columns from data sheet
    tmp = df[[date_col, temp_col]].copy()

    #convert to datetime and delete bad rows
    tmp[date_col] = pd.to_datetime(tmp[date_col], errors="coerce")
    if tmp[date_col].isna().any():
        n_bad = tmp[date_col].isna().sum()
        tmp = tmp.dropna(subset=[date_col])
        print(f"extract_temperature: dropped {n_bad} rows with bad dates")

    #index by time
    tmp = tmp.set_index(date_col).sort_index()

    #ensure numeric temperatures
    tmp[temp_col] = pd.to_numeric(tmp[temp_col], errors="coerce")

    #interpolate small gaps in time
    tmp[temp_col] = tmp[temp_col].interpolate(method="time",
                                              limit=interp_limit)

    #fill any remaining NaN
    if tmp[temp_col].isna().any():
        print("extract_temperature: filling remaining NaNs at edges")
        tmp[temp_col] = tmp[temp_col].fillna(method="bfill").fillna("ffill")

    #return cleaned temperature time series
    ts = tmp[temp_col]
    ts.name = "temperature"
    return ts


# ----------------------------------
# Fourier series
# ----------------------------------
#function that builds fourier approximation, using a specificed number of values k_max
def build_symbolic_fourier(signal_array, sample_rate_hours=1, max_k=None):
    
    #connvert temperature values to flot
    signal = np.asarray(signal_array, dtype=float)
    #signal size (8760hours for year)
    N = signal.size

    #convert hourly timestamp to days
    #dt_days = sample_rate_hours / 24
    #time period (year)
    #T_period = N * dt_days
    T_period = N

    #compute discrete fourier transform (DFT)
    fft_vals = np.fft.fft(signal)
    #divide by N for true Fourier coefficents
    #sympy DFT gives unscaled DFT initially - normalises amplitude to match original data
    Ck = fft_vals / N

    #choose how many frequencies to keep
    #if None will use all frequencies (8760)
    if max_k is None or max_k >= N:
        K = N - 1
    else:
        K = int(max_k)

    #create symbolic varaible t - represnts time (days)
    t_hours = sp.symbols('t', real=True)
    #create empty variable which will store final symbolic equation
    T_sym = 0

    #empty lists for Fourier
    #a is cosine
    #b is sine
    #omega is frequency in rad/day
    a_list = []
    b_list = []
    omega_list = []

    #loop through frequencies k
    for k in range(K + 1):
        C = Ck[k]
        #takes cosine amplitude - real value from array of DFT
        a = float(np.real(C))
        #sine amplitude - imaginary value
        b = float(np.imag(C))
        #compute angular frequency
        omega_k = 2.0 * np.pi * k / T_period

        #store cosine values
        a_list.append(a)
        #store sine values
        b_list.append(b)
        #store omega values
        omega_list.append(omega_k)

        #build symbolic term for frequency
        #cosine_amp*cos(angular_freq*time) - sine_amp*sin(angular_freq*time)
        term = a * sp.cos(omega_k * t_hours) - b * sp.sin(omega_k * t_hours)
        #add term to output
        T_sym += term

    info = {
        "T_period_days": T_period,
        "N": N,
        "a_coeffs": np.array(a_list),
        "b_coeffs": np.array(b_list),
        "omega_k": np.array(omega_list),
    }

    return t_hours, T_sym, info


# ----------------------------------
#main
# ----------------------------------

def main():
    #load data
    df = load_data("EdiTempYear.xlsx", sheet_name="Raw_Data")
    #clean data and output cleaned temp values
    ts = extract_temperature(df, date_col="ob_time",
                             temp_col="air_temperature",
                             interp_limit=6)

    temps = ts.values

    #build symbolic fourier using temp values, sample rate of an hour, max number of terms
    t_sym, T_sym, info = build_symbolic_fourier(
        temps,
        sample_rate_hours=1,
        max_k=150  # None = all harmonics
    )


    #export as txt for later analysis
    root = repo_root()
    out_dir = root / "out"
    out_path = out_dir / "symbolic_fourier_temperature.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(str(T_sym))

    
    print(f"{out_path}")

    #For graph output only
    #convert to numerical
    T_num = sp.lambdify(t_sym, T_sym, "numpy")
    #number of data points
    N = info["N"]
    #number of days - hours -> day
    t_days = np.arange(N) / 24.0
    #evaluate fourier numerically for each day
    recon = T_num(t_days)

    #produce plot of original vs fourier
    plt.figure(figsize=(12, 5))
    plt.plot(ts.index, temps, label="Original")
    plt.plot(ts.index, recon, label="Fourier reconstruction", alpha=0.8)
    plt.legend()
    plt.title("Original vs Symbolic Fourier Reconstruction (numeric check)")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
