# -*- coding: utf-8 -*-
"""
Created on Wed Oct  8 16:06:28 2025

@author: Cailean
"""

# Start
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
def load_data(filename="EdiRadTempJan.xlsx", sheet_name="Simple"):
    base = repo_root()
    data_path = base / "data" / filename

    if not data_path.exists():
        raise FileNotFoundError(f"Could not find {data_path}")

    # Read only the needed sheet
    df = pd.read_excel(data_path, sheet_name=sheet_name)
    return df


def save_plot(fig, name: str):
    out_dir = repo_root() / "out"
    out_dir.mkdir(exist_ok=True)
    fig.savefig(out_dir / f"{name}.png", dpi=300, bbox_inches="tight")


#Define Run Read first 10 lines
def main():
    df = load_data("EdiRadTempJan.xlsx", sheet_name="Simple")
    print("Loaded data sample (first 10 rows):\n")
    print(df.head(20))

    # Convert first column to datetime (optional but helps plotting)
    df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])

    # --- Plot 1: Date vs Air Temperature ---
    fig = plt.figure(figsize=(10, 5))
    plt.plot(df.iloc[:, 0], df.iloc[:, 2], label="Air Temperature (°C)", color="tab:red")
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
    plt.plot(df.iloc[:, 0], df.iloc[:, 1], label="Global Irradiation (W/m²)", color="tab:orange")
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))
    plt.gcf().autofmt_xdate()
    plt.xlabel("Date")
    plt.ylabel("Global Irradiation (W/m²)")
    plt.title("Solar Irradiation Over Time")
    plt.grid(True)
    plt.tight_layout()
    save_plot(fig, "Solar_Irradiation")
    plt.show()

    # --- Plot 3: Comparison of both ---
    fig = plt.figure(figsize=(10, 5))
    plt.plot(df.iloc[:, 0], df.iloc[:, 1], label="Global Irradiation (W/m²)", color="tab:orange")
    plt.plot(df.iloc[:, 0], df.iloc[:, 2], label="Air Temperature (°C)", color="tab:red")
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))
    plt.gcf().autofmt_xdate()
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.title("Comparison: Solar Irradiation and Air Temperature")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    save_plot(fig, "Comparision")
    plt.show()
    
#run
if __name__ == "__main__":
    main()