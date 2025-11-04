# -*- coding: utf-8 -*-
"""
Created on Tue Nov  4 18:42:09 2025

@author: carme
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import brentq
import sys
import seaborn as sns
import plotly.express as px

# --- 1. DEFINE CONSTANTS ---
COMFORT_LOW = 18.0
COMFORT_HIGH = 24.0
TARGET_COMFORT_HOURS = 6500 

# --- 2. DATA PROCESSING FUNCTIONS ---

def load_data():
    """
    Loads the two required input files.
    1. The simulation data (wide format) from Excel
    2. The combinations and cost data from CSV
    """
    print("Loading data...")
    try:
        # Load simulation data: R-values in the first column (index)
        temp_df = pd.read_excel("simulation_results_T_indoor.xlsx", index_col=0)
        # Make sure index is float, not string
        temp_df.index = temp_df.index.astype(float)
        
    except FileNotFoundError:
        print("\n--- ERROR ---")
        print("File 'simulation_results_T_indoor.xlsx' not found.")
        print("Run 'solve_ODE' script first.")
        sys.exit() # Stop the script
    except ImportError:
        print("\n--- ERROR ---")
        print("Module 'openpyxl' not found. You need it to read Excel files.")
        print("Please install it by running: pip install openpyxl")
        sys.exit()

    try:
        # Load cost data: R-value is the key to link
        cost_df = pd.read_csv("combinations_and_costs.csv", index_col='R(m²K/W)')
        cost_df.index = cost_df.index.astype(float)
        
    except FileNotFoundError:
        print("\n--- ERROR ---")
        print("File 'combinations_and_costs.csv' not found.")
    except KeyError:
        print("\n--- ERROR ---")
        print("Could not find column 'R(m²K/W)' in 'combinations_and_costs.csv'.")
        sys.exit()

    return temp_df, cost_df

def calculate_comfort(temp_df):
    """
    Counts comfortable hours for the 'wide' format data.
    This function processes the data row by row (axis=1).
    """
    print("Calculating comfortable hours for each R-value...")
    
    def count_comfort_in_row(row):
        return ((row >= COMFORT_LOW) & (row <= COMFORT_HIGH)).sum()

    comfort_series = temp_df.apply(count_comfort_in_row, axis=1)
    comfort_series.name = "comfortable_hours"
    return comfort_series

def merge_data(comfort_series, cost_df):
    """
    Merges comfort data and cost data using the R-value as the key.
    """
    final_df = cost_df.join(comfort_series, how='inner')
    final_df = final_df.dropna(subset=['comfortable_hours', 'Total_Cost'])
    final_df = final_df.sort_index()
    
    if len(final_df) == 0:
        print("\n--- ERROR ---")
        print("Data merge failed! No R-values matched between your two files.")
        print("Check that the R-values in 'simulation_results_T_indoor.xlsx' (col 1)")
        print("and 'combinations_and_costs.csv' (col 'R(m²K/W)') are identical.")
        sys.exit()
        
    return final_df

# --- 3. NUMERICAL METHODS FUNCTIONS ---

def perform_interpolation(final_df):
    """
    Creates smooth functions for comfort and cost vs. R-value.
    """
    print("Performing interpolation...")
    
    interp_func_comfort = interp1d(
        final_df.index,
        final_df['comfortable_hours'],
        kind='cubic',
        fill_value="extrapolate"
    )
    
    interp_func_cost = interp1d(
        final_df.index,
        final_df['Total_Cost'],
        kind='cubic',
        fill_value="extrapolate"
    )
    
    return interp_func_comfort, interp_func_cost

def solve_root_finding_problem(interp_comfort, interp_cost, r_min, r_max):
    """
    Solves the specific design question using root-finding.
    """
    print(f"\n--- Root-Finding Analysis ---")
    print(f"Design Target: {TARGET_COMFORT_HOURS} comfortable hours.")
    print("Solving for the *exact* R-value needed...")

    def comfort_target_function(r_value):
        return interp_comfort(r_value) - TARGET_COMFORT_HOURS

    try:
        solved_r_value = brentq(comfort_target_function, r_min, r_max)
        solved_cost = interp_cost(solved_r_value)
        
        print("\n---Root-Finding Solution ---")
        print(f"To hit the target, the *exact* R-value required is: {solved_r_value:.3f} m²K/W")
        print(f"The interpolated cost for this R-value is: £{solved_cost:.2f}")
        
        return solved_r_value, solved_cost

    except ValueError:
        print(f"\n--- Root-Finding Error ---")
        print(f"Error: Could not find a solution.")
        print(f"The target of {TARGET_COMFORT_HOURS} hours may be")
        print(f"outside the achievable comfort range of your data.")
        return None, None

# --- 4. PLOTTING FUNCTIONS ---

def create_all_plots(final_df, interp_comfort_func, solved_r_value, solved_cost):
    """
    Generates all the final plots for the design report.
    """
    print("\nGenerating plots...")

    # --- Plot 1: The "Sweet Spot" (4D) ---
    try:
        plt.figure(figsize=(12, 8))
        pur_col_name = 'Polyurethane (PUR)_thickness(m)'
        if pur_col_name not in final_df.columns:
            raise KeyError(f"Column '{pur_col_name}' not in combinations_and_costs.csv")
            
        sizes = final_df[pur_col_name] * 500 + 10

        sc = plt.scatter(
            final_df['Total_Cost'],
            final_df['comfortable_hours'],
            c=final_df.index,
            s=sizes,
            cmap='viridis',
            alpha=0.6,
            edgecolors='k',
            linewidth=0.5
        )

        plt.colorbar(sc, label='R-Value (m²K/W)')
        plt.title("Optimization: Cost vs. Comfort (Passive Model)", fontsize=16)
        plt.xlabel("Total Insulation Cost (£)", fontsize=12)
        plt.ylabel("Annual Comfortable Hours (18-24°C)", fontsize=12)

        for thickness in [0.05, 0.1, 0.15]:
            plt.scatter([], [], s=thickness*500+10, c='gray', alpha=0.6,
                        label=f'{thickness*100:.0f}cm PUR')
        plt.legend(scatterpoints=1, frameon=False, labelspacing=1, title='PUR Thickness')

        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig("sweet_spot_4d_plot.png")
        print("Saved 'sweet_spot_4d_plot.png'")

    except KeyError as e:
        print(f"\nWarning: Could not create 4D plot. {e}")
        print("Creating simpler 3D plot instead.")
        plt.figure(figsize=(10, 6))
        sc = plt.scatter(
            final_df['Total_Cost'],
            final_df['comfortable_hours'],
            c=final_df.index,
            cmap='viridis',
            alpha=0.7,
            edgecolors='k'
        )
        plt.colorbar(sc, label='R-Value (m²K/W)')
        plt.title("Optimization: Cost vs. Comfort (Passive Model)", fontsize=16)
        plt.xlabel("Total Insulation Cost (£)", fontsize=12)
        plt.ylabel("Annual Comfortable Hours (18-24°C)", fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig("sweet_spot_plot.png")
        print("Saved 'sweet_spot_plot.png'")


    # --- Plot 2: The "Recipe" Plot (Interactive Ternary) ---
    try:
        df = final_df.copy()
        eps_col = 'Expanded Polystyrene (EPS)_thickness(m)'
        mw_col = 'mineral_wool_thickness(m)'
        pur_col = 'Polyurethane (PUR)_thickness(m)'
        
        cols = [eps_col, mw_col, pur_col]
        if not all(c in df.columns for c in cols):
            raise KeyError("One or more material thickness columns are missing.")
            
        df['Total_Thickness'] = df[cols].sum(axis=1)

        df['EPS_Frac'] = df[eps_col] / df['Total_Thickness'].replace(0, np.nan)
        df['MW_Frac'] = df[mw_col] / df['Total_Thickness'].replace(0, np.nan)
        df['PUR_Frac'] = df[pur_col] / df['Total_Thickness'].replace(0, np.nan)
        df = df.fillna(0)

        fig = px.ternary(
            df,
            a="EPS_Frac",
            b="MW_Frac",
            c="PUR_Frac",
            color="comfortable_hours",
            size="Total_Cost",
            hover_name=df.index,
            color_continuous_scale='viridis',
            title="Insulation Combination 'Recipe' vs. Comfort"
        )
        fig.update_layout(
            ternary_a_axis_title_text='EPS Fraction',
            ternary_b_axis_title_text='Mineral Wool Fraction',
            ternary_c_axis_title_text='PUR Fraction'
        )
        fig.write_html("ternary_comfort_plot.html")
        print("Saved interactive 'ternary_comfort_plot.html'")
    
    except KeyError as e:
        print(f"\nWarning: Could not create Ternary plot. Missing column: {e}")
    except Exception as e:
        print(f"\nWarning: Could not create Ternary plot. Error: {e}")


    # --- Plot 3: The Correlation Heatmap ---
    try:
        cols_to_correlate = [
            'Expanded Polystyrene (EPS)_thickness(m)', 
            'mineral_wool_thickness(m)', 
            'Polyurethane (PUR)_thickness(m)', 
            'Total_Cost', 
            'comfortable_hours'
        ]
        corr_df = final_df.copy()
        corr_df['R_Value'] = final_df.index
        
        cols_to_correlate = [c for c in cols_to_correlate if c in corr_df.columns]
        corr_df = corr_df[cols_to_correlate + ['R_Value']]

        corr_matrix = corr_df.corr()

        plt.figure(figsize=(10, 7))
        sns.heatmap(
            corr_matrix,
            annot=True,
            cmap='coolwarm',
            fmt=".2f"
        )
        plt.title("Correlation Matrix of Design Variables", fontsize=16)
        plt.tight_layout()
        plt.savefig("correlation_heatmap.png")
        print("Saved 'correlation_heatmap.png'")
        
    except Exception as e:
        print(f"\nWarning: Could not create Heatmap plot. Error: {e}")


    # --- Plot 4: The Root-Finding Validation Plot ---
    plt.figure(figsize=(10, 6))
    
    plt.plot(final_df.index, final_df['comfortable_hours'], 'o', 
             label='Simulated Data Points', markersize=8)
    
    r_smooth = np.linspace(final_df.index.min(), final_df.index.max(), 200)
    comfort_smooth = interp_comfort_func(r_smooth)
    plt.plot(r_smooth, comfort_smooth, '-', label='Interpolated Function (Cubic)')
    
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
    
    # Show all static plots at the end
    plt.show()