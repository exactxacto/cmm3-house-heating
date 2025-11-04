# -*- coding: utf-8 -*-
"""
Created on Tue Nov  4 18:46:17 2025

@author: carme
"""

"""
    It imports all functions from the 'analysis_functions.py' toolbox
"""

# Import the "toolbox" file and give it a short name 'af'
import analysis_functions as af

# --- 5. MAIN EXECUTION ---
if __name__ == "__main__":
    
    # Step 1: Load the two data files
    temp_df, cost_df = af.load_data()
    print(f"Loaded {len(temp_df)} R-value simulations.")
    print(f"Loaded {len(cost_df)} cost combinations.")

    # Step 2: Process the 'wide' temp file to count comfort
    comfort_series = af.calculate_comfort(temp_df)
    
    # Step 3: Merge comfort and cost data
    final_df = af.merge_data(comfort_series, cost_df)
    print(f"Successfully merged {len(final_df)} matching R-values.")
    print("\n--- Final Data Summary (Top 5 rows) ---")
    print(final_df.head())

    # Step 4: Perform Interpolation
    interp_comfort, interp_cost = af.perform_interpolation(final_df)
    
    # Step 5: Solve the Root-Finding Problem
    r_min = final_df.index.min()
    r_max = final_df.index.max()
    solved_r, solved_c = af.solve_root_finding_problem(interp_comfort, interp_cost, r_min, r_max)

    # Step 6: Create and save all final plots
    af.create_all_plots(final_df, interp_comfort, solved_r, solved_c)
    
    print("\nAnalysis complete.")