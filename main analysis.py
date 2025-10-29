# -*- coding: utf-8 -*-
"""
Created on Wed Oct 29 11:23:46 2025

@author: carme
"""
def solve_root_finding_problem(results_df):
    """
    Solves the main design question using interpolation and root-finding
    for ALL materials.
    """
    TARGET_COMFORT_HOURS = 6500
    all_materials = results_df['material_type'].unique()
    solutions = []

    print("\n--- ✅ Root Finding Analysis ---")
    print(f"Design Target: {TARGET_COMFORT_HOURS} comfortable hours.")
    print("Finding cheapest solution for *all* materials...")

    for mat in all_materials:
        # 1. Create interpolation functions for this material
        comfort_func = perform_interpolation(results_df, mat, 'comfortable_hours')
        cost_func = perform_interpolation(results_df, mat, 'total_cost_gbp')
        
        if comfort_func is None or cost_func is None:
            print(f"Skipping {mat} (not enough data).")
            continue

        # 2. Define the root-finding function
        #    We want to find 't' where comfort_func(t) - TARGET = 0
        def comfort_target_function(thickness):
            return comfort_func(thickness) - TARGET_COMFORT_HOURS

        try:
            # 3. Solve for thickness (assuming new 0.4m range)
            #    We search for a root between 0.01m and 0.4m
            min_thickness = brentq(comfort_target_function, 0.01, 0.4)
            
            # 4. Find the cost at that exact thickness
            cost_at_min_thickness = cost_func(min_thickness)
            
            solutions.append({
                'material': mat,
                'required_thickness_m': min_thickness,
                'cost_at_target': cost_at_min_thickness
            })
            
            print(f"  > Solution for {mat}: {min_thickness:.4f} m thickness = £{cost_at_min_thickness:.2f}")

        except ValueError:
            print(f"  > {mat} cannot reach {TARGET_COMFORT_HOURS} hours within 0.4m.")
            solutions.append({
                'material': mat,
                'required_thickness_m': np.nan,
                'cost_at_target': np.inf # Set to infinity (unachievable)
            })

    # --- Find the Best Solution ---
    if not solutions:
        print("No solutions found.")
        return

    solution_df = pd.DataFrame(solutions)
    best_solution = solution_df.loc[solution_df['cost_at_target'].idxmin()]
    
    print("\n--- Root Finding Conclusion ---")
    print(f"The CHEAPEST solution to meet the {TARGET_COMFORT_HOURS} hour target is:")
    print(f"Material: {best_solution['material']}")
    print(f"Required Thickness: {best_solution['required_thickness_m']:.4f} m")
    print(f"Final Cost: £{best_solution['cost_at_target']:.2f}")