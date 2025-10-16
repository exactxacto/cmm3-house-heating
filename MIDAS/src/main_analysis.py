# -*- coding: utf-8 -*-
"""
Created on Wed Oct  8 16:06:28 2025

@author: Cailean
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# Find the repository root
def repo_root() -> Path:
    if "__file__" in globals():
        return Path(__file__).resolve().parent.parent
    else:
        return Path.cwd()
    
    
# Loads Excel data from the 'data' folder and specified sheet 'Simple'.
def load_data(filename="EdiTempYear.xlsx", sheet_name="Raw_Data"):
    base = repo_root()
    data_path = base / "data" / filename

    if not data_path.exists():
        raise FileNotFoundError(f"Could not find {data_path}")

    df = pd.read_excel(data_path, sheet_name=sheet_name)
    return df


def save_plot(fig, name: str):
    out_dir = repo_root() / "out"
    out_dir.mkdir(exist_ok=True)
    fig.savefig(out_dir / f"{name}.png", dpi=300, bbox_inches="tight")


# Detects missing values and replaces them by averaging the top and bottom values
def interpolate_data(df: pd.DataFrame):
    # Loop through each column except the datetime one
    for col in df.columns[1:]:
        # Report how many NaNs before fixing
        missing_count = df[col].isna().sum()
        if missing_count > 0:
            print(f"Interpolating {missing_count} missing values in '{col}'...")
            # Perform linear interpolation (average of top and bottom)
            df[col] = df[col].interpolate(method='linear', limit_direction='both')
    return df


# Define Run Read first 10 lines
def main():
    df = load_data("EdiTempYear.xlsx", sheet_name="Raw_Data")
    print("Loaded data sample (first 10 rows):\n")
    print(df.head(20))

    # Convert first column to datetime
    df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])

    # Clean data by interpolating
    df = interpolate_data(df)

    # --- Plot 1: Date vs Air Temperature ---
    fig = plt.figure(figsize=(10, 5))
    plt.plot(df.iloc[:, 0], df.iloc[:, 1], label="Air Temperature (°C)", color="tab:red")
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))
    plt.gcf().autofmt_xdate()
    plt.xlabel("Date")
    plt.ylabel("Temperature (°C)")
    plt.title("Air Temperature Over Time")
    plt.grid(True)
    plt.tight_layout()
    save_plot(fig, "Air_Temperature")
    plt.show()

    # --- Plot 2: Date vs Solar Irradiation ---
    fig = plt.figure(figsize=(10, 5))
    plt.plot(df.iloc[:, 0], df.iloc[:, 2], label="Global Irradiation (W/m²)", color="tab:orange")
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))
    plt.gcf().autofmt_xdate()
    plt.xlabel("Date")
    plt.ylabel("Global Irradiation (W/m²)")
    plt.title("Solar Irradiation Over Time")
    plt.grid(True)
    plt.tight_layout()
    save_plot(fig, "Solar_Irradiation")
    plt.show()


# run
if __name__ == "__main__":
    main()
