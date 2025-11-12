# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import brentq
import sys
import seaborn as sns
import plotly.express as px

# 1. DEFINE CONSTANTS 
COMFORT_LOW = 18.0
COMFORT_HIGH = 24.0
TARGET_COMFORT_HOURS = 145 

# 2. DATA PROCESSING FUNCTIONS 

def load_data():
    try:
        temp_df = pd.read_excel("simulation_results_T_indoor.xlsx")
        
    except FileNotFoundError:
        print("\n--- ERROR ---")
        print("File 'simulation_results_T_indoor.xlsx' not found.")
        print("Please run the 'solve_ODE' script first.")
        sys.exit() 
    except ImportError:
        print("\n--- ERROR ---")
        print("Module 'openpyxl' not found. Please install: pip install openpyxl")
        sys.exit()

    try:
        cost_df = pd.read_csv("combinations_and_costs.csv")
        
    except FileNotFoundError:
        print("\n--- ERROR ---")
        print("File 'combinations_and_costs.csv' not found.")
        print("Please create this file from the R-value combinations data.")
        sys.exit()

    if len(temp_df) != len(cost_df):
        print("\n--- ERROR ---")
        print(f"File row counts do not match!")
        print(f"Excel has {len(temp_df)} rows.")
        print(f"CSV has {len(cost_df)} rows.")
        print("Please ensure both files were generated from the same list.")
        sys.exit()

    return temp_df, cost_df

def calculate_comfort(temp_df):
    """
    Counts comfortable hours for the 'wide' format data..
    """
    print("Calculating comfortable hours for each R-value...")
    
    temp_data_only = temp_df.drop(columns=temp_df.columns[0])
    
    def count_comfort_in_row(row):
        return ((row >= COMFORT_LOW) & (row <= COMFORT_HIGH)).sum()

    comfort_series = temp_data_only.apply(count_comfort_in_row, axis=1)
    
    comfort_series.name = "comfortable_hours"
    return comfort_series

def merge_data(comfort_series, cost_df):
    """
    Merges comfort data and cost data using the row index.
    """
    final_df = cost_df.join(comfort_series)
    
    final_df = final_df.dropna(subset=['comfortable_hours', 'Total_Cost'])
    
    print("Averaging combinations with duplicate R-values...")
    try:
        r_value_col = 'R(m²K/W)'
        final_df_agg = final_df.groupby(r_value_col).mean(numeric_only=True)
    except KeyError:
        print(f"--- ERROR ---")
        print(f"Cannot find R-value column '{r_value_col}' to group by.")
        sys.exit()

    final_df_agg = final_df_agg.sort_index()
    
    if len(final_df_agg) == 0:
        print("\n--- ERROR ---")
        print("Data merge failed! The files might be empty.")
        sys.exit()
        
    return final_df_agg

# 3. NUMERICAL METHODS FUNCTIONS 

def perform_interpolation(final_df):
    """
    Creates linear interpolation functions for comfort and cost vs. R-value.
    """
    print("Performing interpolation...")
    
    x_axis_r_value = final_df.index
    
    interp_func_comfort = interp1d(
        x_axis_r_value,
        final_df['comfortable_hours'],
        kind='linear',
        fill_value="extrapolate"
    )
    
    interp_func_cost = interp1d(
        x_axis_r_value,
        final_df['Total_Cost'],
        kind='linear',
        fill_value="extrapolate"
    )
    
    return interp_func_comfort, interp_func_cost

def solve_root_finding_problem(interp_comfort, interp_cost, r_min, r_max):
    """
    Solves the specific design question using root-finding.
    """
    print(f"\n--- Root-Finding Analysis ---")
    print(f"Design Target: {TARGET_COMFORT_HOURS} comfortable hours.")
    print("Solving for the required R-value...")

    def comfort_target_function(r_value):
        return interp_comfort(r_value) - TARGET_COMFORT_HOURS

    try:
        solved_r_value = brentq(comfort_target_function, r_min, r_max)
        solved_cost = interp_cost(solved_r_value)
        
        print("\n--- Root-Finding Solution ---")
        print(f"Required R-value: {solved_r_value:.3f} m²K/W")
        print(f"Interpolated cost: £{solved_cost:.2f}")
        
        return solved_r_value, solved_cost

    except ValueError:
        print(f"\n--- Root-Finding Error ---")
        print(f"Error: Could not find a solution.")
        print(f"The target of {TARGET_COMFORT_HOURS} hours may be")
        print(f"outside the achievable comfort range of the data.")
        return None, None

# 4. ANALYSIS FUNCTION 

def find_best_tradeoff(final_df):
    """
    Finds the "knee" of the Cost vs. R-Value curve (best trade-off).
    """
    print("\n--- Design Optimization Analysis ---")
    
    try:
        r_values = final_df.index
        costs = final_df['Total_Cost']
        
        # Normalize data (0 to 1)
        norm_r = (r_values - r_values.min()) / (r_values.max() - r_values.min())
        norm_cost = (costs - costs.min()) / (costs.max() - costs.min())
        
        # Find distance from the "ideal" point (0, 1) [min cost, max R]
        distances = np.sqrt(norm_cost**2 + (1 - norm_r)**2)
        
        # Find the R-value of the point with the minimum distance
        best_r_value = distances.idxmin()
        best_combo = final_df.loc[best_r_value]

        # Print the result
        print("\n--- Optimal Cost-Performance Point ---")
        print(f"   R-Value: {best_combo.name:.3f} m²K/W")
        print(f"   Total Cost: £{best_combo['Total_Cost']:.2f}")
        print(f"   Comfortable Hours: {best_combo['comfortable_hours']:.0f}")
        print("   Averaged Recipe (Total = 0.4m):")
        print(f"    EPS: {best_combo['Expanded Polystyrene (EPS)_thickness(m)']*100:.1f} cm")
        print(f"    MW:  {best_combo['mineral_wool_thickness(m)']*100:.1f} cm")
        print(f"    PUR: {best_combo['Polyurethane (PUR)_thickness(m)']*100:.1f} cm")
        
        return best_combo

    except Exception as e:
        print(f"Could not run optimization analysis: {e}")
        return None


# 5. PLOTTING FUNCTIONS 

def create_all_plots(final_df, interp_comfort_func, solved_r_value, solved_cost, best_combo):
    # --- Plot 1: Cost vs. R-Value Optimization ---
    plt.figure(figsize=(12, 8))
    sc = plt.scatter(
        final_df.index,  # X-axis is R-Value
        final_df['Total_Cost'], # Y-axis is Cost
        c=final_df['comfortable_hours'], # Color is the result
        cmap='viridis',
        alpha=0.6,
        edgecolors='k',
        linewidth=0.5,
        s=50 
    )

    plt.colorbar(sc, label='Comfortable Hours (Annual)')
    plt.title("Optimization: Total Cost vs. R-Value", fontsize=16)
    plt.xlabel("Wall-Only R-Value (m²K/W)", fontsize=12)
    plt.ylabel("Total Insulation Cost (£)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig("cost_vs_rvalue_plot.png")
    print("Saved 'cost_vs_rvalue_plot.png'")


    # Plot 2: Material Fraction Sensitivity 
    print("Generating 'Material Fraction' plots...")
    try:
        df = final_df.copy()
        eps_col = 'Expanded Polystyrene (EPS)_thickness(m)'
        mw_col = 'mineral_wool_thickness(m)'
        pur_col = 'Polyurethane (PUR)_thickness(m)'
        
        cols = [eps_col, mw_col, pur_col]
        if not all(c in df.columns for c in cols):
            raise KeyError("One or more material thickness columns are missing.")
            
        df['Total_Thickness'] = df[cols].sum(axis=1)

        # Calculate fractions
        df['EPS_Frac'] = df[eps_col] / df['Total_Thickness'].replace(0, np.nan)
        df['MW_Frac'] = df[mw_col] / df['Total_Thickness'].replace(0, np.nan)
        df['PUR_Frac'] = df[pur_col] / df['Total_Thickness'].replace(0, np.nan)
        df = df.fillna(0)

        # Create 3 subplots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15), sharex=True)
        
        # Plot 1: EPS Fraction
        s1 = ax1.scatter(df['EPS_Frac'], df['comfortable_hours'], c=df['Total_Cost'], cmap='viridis', alpha=0.7)
        ax1.set_title("Comfort vs. EPS Fraction", fontsize=14)
        ax1.set_ylabel("Comfortable Hours")
        ax1.grid(True, linestyle='--', alpha=0.5)
        
        # Plot 2: Mineral Wool Fraction
        s2 = ax2.scatter(df['MW_Frac'], df['comfortable_hours'], c=df['Total_Cost'], cmap='viridis', alpha=0.7)
        ax2.set_title("Comfort vs. Mineral Wool Fraction", fontsize=14)
        ax2.set_ylabel("Comfortable Hours")
        ax2.grid(True, linestyle='--', alpha=0.5)
        
        # Plot 3: PUR Fraction
        s3 = ax3.scatter(df['PUR_Frac'], df['comfortable_hours'], c=df['Total_Cost'], cmap='viridis', alpha=0.7)
        ax3.set_title("Comfort vs. PUR Fraction", fontsize=14)
        ax3.set_ylabel("Comfortable Hours")
        ax3.set_xlabel("Material Fraction of Total Thickness (0.0 to 1.0)", fontsize=12)
        ax3.grid(True, linestyle='--', alpha=0.5)

        fig.colorbar(s1, ax=ax3, label='Total Cost (£)', orientation='horizontal', pad=0.1)
        plt.tight_layout()
        plt.savefig("material_fraction_plot.png")
        print("Saved 'material_fraction_plot.png'")

    except KeyError as e:
        print(f"\nWarning: Could not create Material Fraction plots. Missing column: {e}")
    except Exception as e:
        print(f"\nWarning: Could not create Material Fraction plots. Error: {e}")


    # Plot 3: The Root-Finding Validation Plot 
    plt.figure(figsize=(10, 6))
    
    x_axis_r_value = final_df.index
    
    plt.plot(x_axis_r_value, final_df['comfortable_hours'], 'o', 
             label='Simulated Data Points (Averaged)', markersize=8)
    
    r_smooth = np.linspace(x_axis_r_value.min(), x_axis_r_value.max(), 200)
    comfort_smooth = interp_comfort_func(r_smooth)
    plt.plot(r_smooth, comfort_smooth, '-', label='Interpolated Function (Linear)')
    
    if solved_r_value:
        plt.axhline(y=TARGET_COMFORT_HOURS, color='red', linestyle='--', 
                    label=f'Target ({TARGET_COMFORT_HOURS} hrs)')
        plt.axvline(x=solved_r_value, color='green', linestyle=':', 
                    label=f'Solution = {solved_r_value:.3f} R-value')
    
    plt.title("Root-Finding: R-Value vs. Comfort", fontsize=16)
    plt.xlabel("R-Value (m²K/W)", fontsize=12)
    plt.ylabel("Annual Comfortable Hours", fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig("root_finding_plot.png")
    print("Saved 'root_finding_plot.png'")
    
    plt.show()

# Plot 4: Time-Series Comparison Plot
def plot_timeseries_comparison(temp_df, cost_df, comfort_series):
    """
    Plots a 3-day sample of indoor temps for the
    best, worst, and median R-value combinations.
    """
    print("Generating time-series comparison plot...")
    
    full_df = cost_df.join(comfort_series)
    
    idx_min = full_df['comfortable_hours'].idxmin()
    idx_max = full_df['comfortable_hours'].idxmax()
    
    median_comfort_val = full_df['comfortable_hours'].median()
    idx_med = (full_df['comfortable_hours'] - median_comfort_val).abs().idxmin()
    
    r_min = full_df.loc[idx_min, 'R(m²K/W)']
    r_max = full_df.loc[idx_max, 'R(m²K/W)']
    r_med = full_df.loc[idx_med, 'R(m²K/W)']
    
    temp_data_only = temp_df.drop(columns=temp_df.columns[0])
    
    series_min = temp_data_only.loc[idx_min]
    series_max = temp_data_only.loc[idx_max]
    series_med = temp_data_only.loc[idx_med]
    
    sample_hours = range(72)
    start_hour = 4000
    end_hour = start_hour + 72
    
    plt.figure(figsize=(15, 7))
    plt.plot(sample_hours, series_min.iloc[start_hour:end_hour], label=f'Worst Comfort ({full_df.loc[idx_min, "comfortable_hours"]} hrs, R={r_min:.2f})', color='blue', alpha=0.7)
    plt.plot(sample_hours, series_med.iloc[start_hour:end_hour], label=f'Median Comfort ({full_df.loc[idx_med, "comfortable_hours"]} hrs, R={r_med:.2f})', color='orange', alpha=0.7, linestyle='--')
    plt.plot(sample_hours, series_max.iloc[start_hour:end_hour], label=f'Best Comfort ({full_df.loc[idx_max, "comfortable_hours"]} hrs, R={r_max:.2f})', color='green', alpha=0.7)
    
    plt.axhline(COMFORT_LOW, color='gray', linestyle=':', label='Comfort Band (18-24°C)')
    plt.axhline(COMFORT_HIGH, color='gray', linestyle=':')
    plt.fill_between(sample_hours, COMFORT_LOW, COMFORT_HIGH, color='green', alpha=0.1)

    plt.title("Time-Series Comparison (72 Hour Summer Sample)", fontsize=16)
    plt.xlabel("Hour of Sample (Summer)", fontsize=12)
    plt.ylabel("Indoor Temperature (°C)", fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig("timeseries_comparison_plot.png")
    print("Saved 'timeseries_comparison_plot.png'")