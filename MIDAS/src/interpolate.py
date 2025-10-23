# -*- coding: utf-8 -*-
"""
Created on Thu Oct 16 12:13:04 2025

@author: Cailean
"""
#--------------------------
#Setup and File Find
#--------------------------

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks

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

#--------------------------
#Define Functions
#--------------------------

#Extract temperature data
#Convert date column to Pandas date format 
#Ensure correct data format and order
#Convert time to numeric value
#Clean Data - Fill in NaN values
def extract_temperature(df, date_col="ob_time", temp_col="air_temperature", interp_limit=6):
    
    if date_col not in df.columns:
        raise KeyError(f"Date column {date_col} not found in DataFrame")
        
    #Make copy of DataFrame
    tmp = df[[date_col, temp_col]].copy()
    
    #Date column to pandas datetime
    tmp[date_col] = pd.to_datetime(tmp[date_col], errors="coerce")
    
    #Remove any bad rows, inform which were removed
    if tmp[date_col].isna().any():
        n_bad = tmp[date_col].isna().sum()
        tmp = tmp.dropna(subset=[date_col])
        print(f"extract_temperature: dropped {n_bad} rows with {date_col}")
       
    #Set Index and sort by time
    tmp = tmp.set_index(date_col).sort_index()
    
    #Check Temp is numeric - NaN if not
    tmp[temp_col] = pd.to_numeric(tmp[temp_col], errors="coerce")
    
    #Interpolate small gaps - Leave larger gaps for accuracy
    tmp[temp_col] = tmp[temp_col].interpolate(method="time", limit=interp_limit)
    
    #Fill any remaining NaN remaining
    if tmp[temp_col].isna().any():
        print(f"extract_temperature: filling {tmp[temp_col].isna().sum()} edge NaNs with nearest valid values")
        tmp[temp_col] = tmp[temp_col].fillna(method="bfill").fillna(method="ffill")
    
    #Convert to series for return value
    ts = tmp[temp_col]
    ts.name = "temperature"
    
    return ts


#Function to remove moving trend from data
#Creates series placed around 0, just fluctations modelled
#Forier constructed around fluctuations and trend

def detrend_temperature(df, temp_col="air_temperature", window_hours=24*30):
    #Make copy of data
    df_out = df.copy()
    
    #Compute trend (Slow Mean)
    df_out["Trend"] = df_out[temp_col].rolling(window=window_hours, center=True, min_periods=1).mean()
    
    #Removes slow trend from data
    df_out["Detrended"] = df_out[temp_col] - df_out["Trend"]
    df_out["Detrended"] = df_out["Detrended"].fillna(method="bfill").fillna(method="ffill")
    
    return df_out


#Compute and plot the FFT of the temperature data
#Fast Fourier Transform
#Frequency in days not hours
def compute_fft(df, col="Detrended", sample_rate_hours=1):
    #df - pd.DataFrame
    #col - Column name to transform
    #sample_rate_hours - Time between samples
    
    #Convert Detrended temperature values to numpy array
    signal = df[col].to_numpy()
    
    #N - Number of values in data set
    N = len(signal)
    
    #Performs FFT - Converts time domain data to frequency and amplitude domain
    fft_vals = np.fft.fft(signal)
    
    #Calculates coresponding frequency for each hour
    freqs_hour = np.fft.fftfreq(N, d=sample_rate_hours)
    
    #Convert to days
    freqs_day = freqs_hour * 24
    
    #Calculates amplitude specturm
    #np.abs() - absolute value of amplitudes
    #(2/N) - Only takes into account positive values as is symetric
    amplitudes = (2/N) * np.abs(fft_vals)
    
    #Ensures no negitive values
    mask = freqs_day > 0
    freqs_day = freqs_day[mask]
    amplitudes = amplitudes[mask]
    
    #Plot Freq and Amplitude
    plt.figure(figsize=(10,5))
    plt.plot(freqs_day, amplitudes)
    plt.title("FFT Spectrum of Detrended Temperature")
    plt.xlabel("Frequency (cycles per day)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    return freqs_day, amplitudes

#Identiify the dominant frequencies in the FFT spectrum
def identify_main_frequencies(freqs_day, amplitudes, n_peaks=5, height_ratio=0.05):
    #Find all peaks - Only detect local maxima with amp greater than height ratio
    peaks, props = find_peaks(amplitudes, height=np.max(amplitudes)*height_ratio)
    
    #Sort peaks in decending order
    top_peaks = peaks[np.argsort(props['peak_heights'])[::-1][:n_peaks]]
    
    #Extracts freq and amp from peaks
    dom_freqs = freqs_day[top_peaks]
    dom_amps = amplitudes[top_peaks]
    
    #Plot Freq and Amplitude with peaks
    plt.figure(figsize=(10,5))
    plt.plot(freqs_day, amplitudes, label='FFT Spectrum')
    plt.scatter(dom_freqs, dom_amps, color='red', label='Dominant Peaks')
    plt.title("Dominant Frequencies in Temperature Signal")
    plt.xlabel("Frequency (cycles per day)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    return dom_freqs, dom_amps, top_peaks

#Reconstruct Forier using only dominant frequencys
#Filtering to show only main patterns
def reconstruct_from_peaks(signal_array, freqs_day_pos, dom_freqs, dom_amps, peak_indices,
                           dt_hours=1, add_trend=None, detrended=None):
    
    #Detrended numby array
    signal = np.asarray(signal_array)
    N = len(signal)

    #Recompute full FFT
    #Convvert to cycles per day
    fft_full = np.fft.fft(signal)
    freqs_hour_full = np.fft.fftfreq(N, d=1.0)
    freqs_day_full = freqs_hour_full * 24.0

    #Keeps only positive values from FFT
    pos_mask = freqs_day_full > 0
    fft_pos = fft_full[pos_mask]
    freqs_day_pos_recomp = freqs_day_full[pos_mask]

    #Stores indices of FFT bins used in reconstruction
    used_bins = []
    #Store phase of each frequency
    phases = []
    #Set base reconstruct all 0
    recon = np.zeros(N, dtype=float)
    #time array in days, used for computing cosine waves
    t_days = np.arange(N) * dt_hours / 24.0

    #Loops over each dominant freqeuncy - Picks closest FFt bin to that frequency
    #Extracts FFT value and computes phase
    #Adds cosine to aassigned amplitude and phase
    for f, A in zip(dom_freqs, dom_amps):
        idx_pos = int(np.argmin(np.abs(freqs_day_pos_recomp - f)))
        used_bins.append(idx_pos)

        complex_val = fft_pos[idx_pos]
        phi = np.angle(complex_val)
        phases.append(phi)

        recon += A * np.cos(2.0 * np.pi * f * t_days + phi)

    #Add trend/mean back if want
    if add_trend is not None:
        recon = recon + np.asarray(add_trend)

    #Compute RMSE if detrended provided
    if detrended is not None:
        detrended = np.asarray(detrended)
        rmse = float(np.sqrt(np.mean((detrended - recon)**2)))
    else:
        rmse = None

    return recon, rmse, np.array(phases), np.array(used_bins)
  

def rmse_vs_n_peaks(signal, freqs_day_pos, amplitudes, trend=None, n_peaks_list=None, dt_hours=1, height_ratio=0.05):

    if n_peaks_list is None:
        n_peaks_list = [3, 6, 10, 15, 25, 40]

    rmse_list = []
    recon_list = []

    for n_peaks in n_peaks_list:
        dom_freqs, dom_amps, peak_idxs = identify_main_frequencies(
            freqs_day_pos, amplitudes, n_peaks=n_peaks, height_ratio=height_ratio
        )
        recon, rmse, phases, used_bins = reconstruct_from_peaks(
            signal,
            freqs_day_pos,
            dom_freqs,
            dom_amps,
            peak_idxs,
            dt_hours=dt_hours,
            add_trend=trend,
            detrended=signal
        )
        rmse_list.append(rmse)
        recon_list.append(recon)

    return n_peaks_list, rmse_list, recon_list
        

#--------------------------
#Define Main
#--------------------------

#Example Main
def main():
    df = load_data("EdiTempYear.xlsx", sheet_name="Raw_Data")
    print("Loaded data sample (first 10 rows):\n")
    print(df.head(10))
    
    
    df = detrend_temperature(df, temp_col="air_temperature", window_hours=24*30)
    
    plt.figure(figsize=(12,6))
    
    sample = df.iloc[:24*30*2]
    
    plt.plot(sample["ob_time"], sample["air_temperature"], label="Original Temp", alpha=0.6)
    plt.plot(sample["ob_time"], sample["Trend"], label="Slow Trend (30d avg)", linewidth=2)
    plt.plot(sample["ob_time"], sample["Detrended"], label="Detrended Temp", alpha=0.8)
    
    plt.title("Temperature Detrending Check")
    plt.xlabel("Date")
    plt.ylabel("Temperature (°C)")
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    freqs_day_pos, amplitudes = compute_fft(df, col="Detrended", sample_rate_hours=1)
    dom_freqs, dom_amps, peak_idxs = identify_main_frequencies(freqs_day_pos, amplitudes, n_peaks=8, height_ratio=0.05)
    
    signal_array = df["Detrended"].values
    recon, rmse, phases, used_bins = reconstruct_from_peaks(
        signal_array,
        freqs_day_pos,
        dom_freqs,
        dom_amps,
        peak_idxs,
        dt_hours=1,
        add_trend=df["Trend"].values if "Trend" in df.columns else None,
        detrended=df["Detrended"].values
        )
    
    print("Reconstruction RMSE (detrended):", rmse)
    print("Used positive-bin indices:", used_bins)
    print("Phases (rad):", phases)
    
    plt.figure(figsize=(12,5))
    plt.plot(df.index, df["air_temperature"], label="Original")
    plt.plot(df.index, recon, label="Reconstructed (Fourier + trend)", alpha=0.9)
    plt.legend()
    plt.title("Original vs Reconstructed Temperature")
    plt.tight_layout()
    plt.show()
    
    signal_array = df["Detrended"].values
    trend = df["Trend"].values if "Trend" in df.columns else None
    
    n_peaks_list, rmse_list, recon_list = rmse_vs_n_peaks(
        signal_array, freqs_day_pos, amplitudes,
        trend=trend,
        n_peaks_list=[3, 6, 10, 15, 25],
        dt_hours=1
        )   
    
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df["air_temperature"], label="Original")
    
    for n_peaks, recon in zip(n_peaks_list, recon_list):
        plt.plot(recon, label=f'{n_peaks} peaks', alpha=0.4)
        
    plt.title("Original vs Reconstructed Signals (Different N Peaks)")
    plt.xlabel("Time (hours)")
    plt.ylabel("Signal amplitude")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # ---- 2. Plot RMSE comparison ----
    plt.figure(figsize=(8, 5))
    plt.plot(n_peaks_list, rmse_list, marker='o', color='blue', linewidth=2)
    plt.title("RMSE vs Number of Peaks")
    plt.xlabel("Number of Peaks")
    plt.ylabel("RMSE")
    plt.grid(True)
    plt.show()
    
    
#--------------------------
# run
#--------------------------
    
if __name__ == "__main__":
    main()
