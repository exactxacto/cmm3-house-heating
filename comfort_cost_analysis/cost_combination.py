# -*- coding: utf-8 -*-
"""
Conbinations and Costs 
"""

import pandas as pd
import sys
import numpy as np 
from pathlib import Path

# Costs per cubic meter (£/m³) 
COSTS = {
    'EPS': 70.0,
    'MW': 60.0,
    'PUR': 150.0
}

# House dimensions 
HOUSE_WIDTH = 9.0   
HOUSE_LENGTH = 4.0
HOUSE_HEIGHT = 3.0
WALL_AREA = 2 * HOUSE_HEIGHT * (HOUSE_WIDTH + HOUSE_LENGTH) # 78.0 m²

script_dir = Path(__file__).resolve().parent   
root = script_dir.parent         

input_file = root / 'thermal_resistance' / 'out' / 'R_results_Expanded Polystyrene (EPS)_mineral_wool_Polyurethane (PUR).txt'
output_file = root / 'combinations_and_costs.csv'


try:
    df = pd.read_csv(input_file, sep='\t', encoding='latin-1')
    print(f"Successfully loaded '{input_file}'.")

except FileNotFoundError:
    print(f"--- ERROR ---")
    print(f"Could not find the input file: '{input_file}'")
    print("Please make sure this file is in the same folder.")
    sys.exit()
except Exception as e:
    print(f"An unexpected error occurred: {e}")
    sys.exit()

eps_col = 'Expanded Polystyrene (EPS)_thickness(m)'
mw_col = 'mineral_wool_thickness(m)'
pur_col = 'Polyurethane (PUR)_thickness(m)'
r_col_bad = 'R(mÂ²K/W)'
r_col_good = 'R(m²K/W)'

if r_col_bad in df.columns:
    print(f"Found weird R-value column '{r_col_bad}', renaming to '{r_col_good}'...")
    df = df.rename(columns={r_col_bad: r_col_good})
else:
    # This is a fallback in case the file was read correctly
    print(f"Could not find column '{r_col_bad}'. Looking for '{r_col_good}'...")

# CHeck
required_cols = [eps_col, mw_col, pur_col, r_col_good]
if not all(col in df.columns for col in required_cols):
    print("--- ERROR ---")
    print("The input file is missing one or more required columns.")
    print("It must contain:")
    print(required_cols)
    print("\nFound columns:")
    print(df.columns.tolist())
    sys.exit()

# Calculate Cost

print(f"Calculating total cost for {len(df)} combinations...")
print(f"Using Wall Area: {WALL_AREA:.2f} m²")

# Calculate the Total_Cost for each combination
df['Total_Cost'] = (
    (df[eps_col] * COSTS['EPS']) +
    (df[mw_col] * COSTS['MW']) +
    (df[pur_col] * COSTS['PUR'])
) * WALL_AREA

try:
    df.to_csv(output_file, index=False)
    print(f"\nSuccessfully created '{output_file}'!")
    print("This file now matches your new R-values.")
    print("You can now run 'main_analysis.py'.")

except PermissionError:
    print(f"\n--- ERROR ---")
    print(f"Could not save '{output_file}'.")
    print("Is the file open in Excel? Close it and try again.")
except Exception as e:
    print(f"\nAn unexpected error occurred: {e}")