"""
PURPOSE:
    Analyzes the 'wide' format temperature simulation data from the team.
    This script is for a PASSIVE model (no heater).
    
    1.  Loads 'all_temperatures.csv' (R-values as rows, 8760 time columns).
    2.  Loads 'combinations_and_costs.csv' (R-value vs. Cost).
    3.  Counts comfortable hours (18-24°C) for each R-value.
    4.  Performs INTERPOLATION to create smooth functions for comfort and cost.
    5.  Uses ROOT-FINDING to find the cheapest R-value for a specific comfort target.
    6.  Generates the final Cost vs. Comfort "sweet spot" plot.

REQUIREMENTS:
    pip install numpy pandas matplotlib scipy seaborn plotly
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
# This is our specific design target for the root-finding problem
TARGET_COMFORT_HOURS = 6500 

# --- 2. DATA PROCESSING FUNCTIONS ---

def load_data():
    """
    Loads the two required input files.
    1. The simulation data (wide format)
    2. The combinations and cost data
    """
    print("Loading data...")
    try:
        # Load simulation data: R-values in the first column (index)
        temp_df = pd.read_excel("simulation_results_T_indoor.xlsx", index_col=0)
        # Make sure index is float, not string
        temp_df.index = temp_df.index.astype(float)
        
    except FileNotFoundError:
        print("\n--- ERROR ---")
        print("File 'all_temperatures.csv' not found.")
        print("Please get this file from Person B and place it in the same folder.")
        sys.exit() # Stop the script

    try:
        # Load cost data: R-value is the key to link
        # We assume the column is named 'R(m²K/W)' from your example
        cost_df = pd.read_csv("combinations_and_costs.csv", index_col='R(m²K/W)')
        cost_df.index = cost_df.index.astype(float)
        
    except FileNotFoundError:
        print("\n--- ERROR ---")
        print("File 'combinations_and_costs.csv' not found.")
        print("Please create this file (from Person A's data + your cost calcs).")
        sys.exit()
    except KeyError:
        print("\n--- ERROR ---")
        print("Could not find column 'R(m²K/W)' in 'combinations_and_costs.csv'.")
        print("Please make sure the R-value column has that *exact* name.")
        sys.exit()

    return temp_df, cost_df

def calculate_comfort(temp_df):
    """
    Counts comfortable hours for the 'wide' format data.
    
    This function processes the data row by row (axis=1).
    """
    print("Calculating comfortable hours for each R-value...")
    
    def count_comfort_in_row(row):
        # This is a fast way to count all values in the row (8760 cols)
        # that are between the two comfort bounds.
        return ((row >= COMFORT_LOW) & (row <= COMFORT_HIGH)).sum()

    # 'axis=1' tells pandas to apply the function to each ROW
    comfort_series = temp_df.apply(count_comfort_in_row, axis=1)
    
    # Rename the series for clarity
    comfort_series.name = "comfortable_hours"
    return comfort_series

def merge_data(comfort_series, cost_df):
    """
    Merges comfort data and cost data using the R-value as the key.
    """
    # Merge the 'cost_df' with the new 'comfort_series'
    # 'how='inner'' means we only keep R-values that are in BOTH files
    final_df = cost_df.join(comfort_series, how='inner')
    
    # Drop any rows that failed (e.g., if cost was missing)
    final_df = final_df.dropna(subset=['comfortable_hours', 'Total_Cost'])
    
    # IMPORTANT: Sort by R-value (the index) to prepare for interpolation
    final_df = final_df.sort_index()
    
    if len(final_df) == 0:
        print("\n--- ERROR ---")
        print("Data merge failed! No R-values matched between your two files.")
        print("Check that the R-values in 'all_temperatures.csv' (col 1)")
        print("and 'combinations_and_costs.csv' (col 'R(m²K/W)') are identical.")
        sys.exit()
        
    return final_df

# --- 3. NUMERICAL METHODS FUNCTIONS ---

def perform_interpolation(final_df):
    """
    Creates smooth functions for comfort and cost vs. R-value.
    [cite_start]This is one of the required numerical methods. [cite: 36, 68]
    """
    print("Performing interpolation...")
    
    # Create a function: comfort = f(r_value)
    interp_func_comfort = interp1d(
        final_df.index,  # X-values (R-values)
        final_df['comfortable_hours'],  # Y-values (Comfort)
        kind='cubic',    # Use a smooth curve
        fill_value="extrapolate" # Allows us to guess slightly outside our data
    )
    
    # Create a function: cost = f(r_value)
    interp_func_cost = interp1d(
        final_df.index,  # X-values (R-values)
        final_df['Total_Cost'],  # Y-values (Cost)
        kind='cubic',
        fill_value="extrapolate"
    )
    
    return interp_func_comfort, interp_func_cost

def solve_root_finding_problem(interp_comfort, interp_cost, r_min, r_max):
    """
    Solves the specific design question using root-finding.
    [cite_start]This is one of the required numerical methods. [cite: 36, 68]
    """
    print(f"\n--- Root-Finding Analysis ---")
    print(f"Design Target: {TARGET_COMFORT_HOURS} comfortable hours.")
    print("Solving for the *exact* R-value needed...")

    # We need to find the root of the function: f(r) = comfort_function(r) - Target
    def comfort_target_function(r_value):
        return interp_comfort(r_value) - TARGET_COMFORT_HOURS

    try:
        # 'brentq' is a fast, reliable root-finding algorithm
        # It searches for a zero-crossing between r_min and r_max
        solved_r_value = brentq(comfort_target_function, r_min, r_max)
        
        # Now that we have the exact R-value, find its cost
        solved_cost = interp_cost(solved_r_value)
        
        print("\n--- Root-Finding Solution ---")
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
    [cite_start]Generates all the final plots for the design report. [cite: 308]
    """
    print("\nGenerating plots...")

    # --- Plot 1: The "Sweet Spot" (4D) ---
    # This is the most important plot for the Cost vs. Comfort "sweet spot"
    try:
        plt.figure(figsize=(12, 8))

        # We use 'PUR_thickness(m)' to scale the size.
        # Add a small value to avoid zero-sized points.
        # Ensure the column name matches your 'combinations_and_costs.csv'
        pur_col_name = 'Polyurethane (PUR)_thickness(m)'
        if pur_col_name not in final_df.columns:
            raise KeyError(f"Column '{pur_col_name}' not in combinations_and_costs.csv")
            
        sizes = final_df[pur_col_name] * 500 + 10

        sc = plt.scatter(
            final_df['Total_Cost'],
            final_df['comfortable_hours'],
            c=final_df.index,  # The index is the R-value
            s=sizes,
            cmap='viridis',    # Colormap (blue-green-yellow)
            alpha=0.6,
            edgecolors='k',
            linewidth=0.5
        )

        plt.colorbar(sc, label='R-Value (m²K/W)')
        plt.title("Optimization: Cost vs. Comfort (Passive Model)", fontsize=16)
        plt.xlabel("Total Insulation Cost (£)", fontsize=12)
        plt.ylabel("Annual Comfortable Hours (18-24°C)", fontsize=12)

        # Create a custom legend for the sizes
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
        # Fallback to a simpler plot if the column is missing
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
    # This shows the 3-material combination recipe
    try:
        df = final_df.copy() # Make a copy
        
        # --- IMPORTANT ---
        # Make sure these column names *exactly* match your CSV file
        eps_col = 'Expanded Polystyrene (EPS)_thickness(m)'
        mw_col = 'mineral_wool_thickness(m)'
        pur_col = 'Polyurethane (PUR)_thickness(m)'
        
        cols = [eps_col, mw_col, pur_col]
        # Check if all columns exist
        if not all(c in df.columns for c in cols):
            raise KeyError("One or more material thickness columns are missing.")
            
        df['Total_Thickness'] = df[cols].sum(axis=1)

        # Calculate proportions (handling zero division)
        df['EPS_Frac'] = df[eps_col] / df['Total_Thickness'].replace(0, np.nan)
        df['MW_Frac'] = df[mw_col] / df['Total_Thickness'].replace(0, np.nan)
        df['PUR_Frac'] = df[pur_col] / df['Total_Thickness'].replace(0, np.nan)
        df = df.fillna(0) # Fill in any NaNs with 0

        fig = px.ternary(
            df,
            a="EPS_Frac",
            b="MW_Frac",
            c="PUR_Frac",
            color="comfortable_hours",
            size="Total_Cost",
            hover_name=df.index, # Shows R-value on hover
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
    # This shows how all the variables relate to each other
    try:
        cols_to_correlate = [
            'Expanded Polystyrene (EPS)_thickness(m)', 
            'mineral_wool_thickness(m)', 
            'Polyurethane (PUR)_thickness(m)', 
            'Total_Cost', 
            'comfortable_hours'
        ]
        # We also add the R-value (the index)
        corr_df = final_df.copy()
        corr_df['R_Value'] = final_df.index
        
        # Keep only the columns that actually exist
        cols_to_correlate = [c for c in cols_to_correlate if c in corr_df.columns]
        corr_df = corr_df[cols_to_correlate + ['R_Value']]

        # Calculate the correlation matrix
        corr_matrix = corr_df.corr()

        plt.figure(figsize=(10, 7))
        sns.heatmap(
            corr_matrix,
            annot=True,     # Show the numbers in the squares
            cmap='coolwarm',# Red (positive) to Blue (negative)
            fmt=".2f"       # Format numbers to 2 decimal places
        )
        plt.title("Correlation Matrix of Design Variables", fontsize=16)
        plt.tight_layout()
        plt.savefig("correlation_heatmap.png")
        print("Saved 'correlation_heatmap.png'")
        
    except Exception as e:
        print(f"\nWarning: Could not create Heatmap plot. Error: {e}")


    # --- Plot 4: The Root-Finding Validation Plot (from before) ---
    plt.figure(figsize=(10, 6))
    
    # Plot the original data points
    plt.plot(final_df.index, final_df['comfortable_hours'], 'o', 
             label='Simulated Data Points', markersize=8)
    
    # Plot the smooth interpolated function
    r_smooth = np.linspace(final_df.index.min(), final_df.index.max(), 200)
    comfort_smooth = interp_comfort_func(r_smooth)
    plt.plot(r_smooth, comfort_smooth, '-', label='Interpolated Function (Cubic)')
    
    # 

    # Plot the target line and the solution
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

# --- 5. MAIN EXECUTION ---

if __name__ == "__main__":
    
    # Step 1: Load the two data files
    temp_df, cost_df = load_data()
    print(f"Loaded {len(temp_df)} R-value simulations.")
    print(f"Loaded {len(cost_df)} cost combinations.")

    # Step 2: Process the 'wide' temp file to count comfort
    comfort_series = calculate_comfort(temp_df)
    
    # Step 3: Merge comfort and cost data
    final_df = merge_data(comfort_series, cost_df)
    print(f"Successfully merged {len(final_df)} matching R-values.")
    print("\n--- Final Data Summary (Top 5 rows) ---")
    print(final_df.head())

    # Step 4: Perform Interpolation
    interp_comfort, interp_cost = perform_interpolation(final_df)
    
    # Step 5: Solve the Root-Finding Problem
    r_min = final_df.index.min()
    r_max = final_df.index.max()
    solved_r, solved_c = solve_root_finding_problem(interp_comfort, interp_cost, r_min, r_max)

    # Step 6: Create and save all final plots
    create_all_plots(final_df, interp_comfort, solved_r, solved_c)
    
    print("\nAnalysis complete.")