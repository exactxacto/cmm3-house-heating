# -*- coding: utf-8 -*-
"""
PURPOSE:
    This is the main script to run the *entire* analysis.
    It imports all functions from the 'analysis_functions.py' toolbox
    and runs them in a clear, step-by-step workflow.
"""

# Import the "toolbox" file and give it a short name 'af'
import analysis_functions as af

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    
    # Step 1: Load the two data files
    temp_df, cost_df = af.load_data()
    print(f"Loaded {len(temp_df)} R-value simulations.")
    print(f"Loaded {len(cost_df)} cost combinations.")

    # Step 2: Process the 'wide' temp file to count comfort
    comfort_series = af.calculate_comfort(temp_df)
    
    # Step 3: Merge comfort and cost data (and average duplicates)
    final_df = af.merge_data(comfort_series, cost_df)
    print(f"Successfully merged {len(final_df)} unique R-values.")
    print("\n--- Final Data Summary (Top 5 rows) ---")
    print(final_df.head())

    # --- NEW STEP 3.5: Find Best Trade-Off ---
    best_combo = af.find_best_tradeoff(final_df)

    # Step 4: Perform Interpolation
    interp_comfort, interp_cost = af.perform_interpolation(final_df)
    
    # Step 5: Solve the Root-Finding Problem
    r_min = final_df.index.min()
    r_max = final_df.index.max()
    solved_r, solved_c = af.solve_root_finding_problem(interp_comfort, interp_cost, r_min, r_max)

    # Step 6: Create and save the "Sweet Spot" and "Root-Finding" plots
    af.create_all_plots(final_df, interp_comfort, solved_r, solved_c, best_combo)
    
    # Step 7: Create the "Best vs. Worst" time-series plot
    af.plot_timeseries_comparison(temp_df, cost_df, comfort_series)
    
    print("\nAnalysis complete.")