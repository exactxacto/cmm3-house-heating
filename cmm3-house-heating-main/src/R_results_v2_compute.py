# -*- coding: utf-8 -*-
"""
Created on Wed Nov  5 11:51:35 2025

@author: carme
"""
#-------------------------------------------------------------------

from pathlib import Path
import matplotlib.pyplot as plt
import itertools
import numpy as np

#find repo root (base folder)
def repo_root() -> Path:
    if "__file__" in globals():
        return Path(__file__).resolve().parent.parent
    else:
        return Path.cwd()

#-------------------------------------------------------------------

"Function to read txt data from materials folder, looks for files containing name,k values"
"Reads txt files"
"Extracts k and name"
"stores in dictionary and creates global variables"
def load_materials(materials_folder="materials"):
    #locate materials folder root/materials
    base = repo_root()
    folder_path = base / materials_folder

    #throws error if folder doesnt exist
    if not folder_path.exists():
        raise FileNotFoundError(f"Could not find materials folder: {folder_path}")

    #empty dictionary for material data
    materials_data = {}

    #loops through all txt files and looks for 'name =' and 'k ='
    for txt_file in folder_path.glob("*.txt"):
        name_val = None
        k_val = None

        #opens txt locates name and k value, strips any spaces
        #stores as name and float
        with open(txt_file, "r") as f:
            for line in f:
                line = line.strip()
                if line.startswith("name ="):
                    _, value = line.split("=", 1)
                    name_val = value.strip()
                elif line.startswith("k ="):
                    _, value = line.split("=", 1)
                    k_val = float(value.strip())

        #adds material to data if k and name is found (requirement)
        if name_val is not None and k_val is not None:
            materials_data[name_val] = k_val
            # Create a global variable with the material name
            var_name = name_val.replace(" ", "_")
            globals()[var_name] = k_val

    #store materials dictionary globally for compute_R()
    globals()["MATERIALS"] = materials_data
    return materials_data

#-------------------------------------------------------------------

"Compute total thermal resistance (R) from variable arguments"
"Materials are looked up in global materials dictionary and calculated"
"Thickness is assumed to be meters"
def compute_R(*args):
    #error if materials dictionary not loaded
    if "MATERIALS" not in globals():
        raise RuntimeError("MATERIALS dictionary not loaded. Run load_materials() first.")

    #error if calculation doesn't contain name and thickness
    if len(args) % 2 != 0:
        raise ValueError("Arguments must be in pairs: (material, thickness)")
    
    #empty float
    total_R = 0.0
    
    #goes through each pair (material and thickness) and calculates r value
    for i in range(0, len(args), 2):
        material = args[i]
        thickness = args[i + 1]

        if material not in MATERIALS:
            raise ValueError(f"Material '{material}' not found in MATERIALS")

        #summation of r values for t_total
        k = MATERIALS[material]
        total_R += thickness / k

    return total_R

#-------------------------------------------------------------------

# *** MODIFIED FUNCTION ***
# This function replaces the old 'sweep_combinations'
# It generates combinations that sum to a fixed total thickness.

def sweep_fixed_total_thickness(combo, total_thickness, step_size=0.01, out_folder="out"):
    """
    Sweeps combinations of materials for a FIXED total thickness.

    combo: tuple of material names (e.g., ("EPS", "Wool", "PUR"))
    total_thickness: the target sum for all thicknesses (e.g., 0.4)
    step_size: the smallest increment of thickness (e.g., 0.01 for 1cm)
    out_folder: folder for output files
    """
    
    if "MATERIALS" not in globals():
        raise RuntimeError("MATERIALS not loaded. Run load_materials() first.")

    # --- 1. Generate the thickness combinations ---
    
    # Calculate the total number of "steps"
    # We round to handle floating point errors
    num_steps = int(round(total_thickness / step_size))
    
    # Check if total_thickness is a multiple of step_size
    if not np.isclose(total_thickness, num_steps * step_size):
        raise ValueError(f"total_thickness ({total_thickness}) is not an even multiple of step_size ({step_size})")

    num_materials = len(combo)
    
    # We find all integer tuples (n1, n2, n3, ...) that sum to 'num_steps'
    # This generates all combinations of (0, 0, ..., num_steps) up to (num_steps, 0, ..., 0)
    # and filters for the ones that sum to the total.
    
    # Create a range of possible steps (0 to num_steps)
    step_range = range(num_steps + 1)
    
    # Generate all products and filter for the correct sum
    all_step_combos = [
        tup for tup in itertools.product(step_range, repeat=num_materials)
        if sum(tup) == num_steps
    ]
    
    # Convert step counts (e.g., 20) back to thicknesses (e.g., 0.4)
    all_thickness_combos = []
    for step_tup in all_step_combos:
        thickness_tup = tuple(n * step_size for n in step_tup)
        all_thickness_combos.append(thickness_tup)
        
    print(f"Generated {len(all_thickness_combos)} combinations that sum to {total_thickness:.2f} m.")

    # --- 2. Calculate R-Values ---
    
    #empty list for results
    results = []

    #loops through all combinations and computes sum of R
    for thicknesses in all_thickness_combos:
        #build argument list for compute_R
        args = []
        for m, t in zip(combo, thicknesses):
            args.extend([m, t])

        R_total = compute_R(*args)
        results.append((*thicknesses, R_total))

    # --- 3. Save Results to File ---
    
    #outfolder is in root folder
    out_path = repo_root() / out_folder
    out_path.mkdir(parents=True, exist_ok=True)

    #creates name based on combination of results
    txt_file = out_path / f"R_results_{'_'.join(combo)}.txt"
    with open(txt_file, "w") as f:
        header = "\t".join([f"{m}_thickness(m)" for m in combo]) + "\tR(m²K/W)\n"
        f.write(header)
        #creates tab seperated txt file with results
        for r in results:
            line = "\t".join([f"{x:.3f}" for x in r]) + "\n"
            f.write(line)

    # --- 4. Plot Graphs ---
    plt.figure(figsize=(8, 5))

    #if 1 material sweep create 2d Line plot
    if len(combo) == 1:
        # This case doesn't make much sense for *fixed* total thickness
        # but we'll leave it
        x = [r[0] for r in results]
        y = [r[1] for r in results]
        plt.plot(x, y, marker="o")
        plt.xlabel(f"{combo[0]} Thickness (m)")
    
    #if 2 materials create 3d graph
    elif len(combo) == 2:
        # This is also less common, as t1 + t2 = 0.4 is just a 2D line
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")

        t1 = np.array([r[0] for r in results])
        t2 = np.array([r[1] for r in results])
        R_vals = np.array([r[2] for r in results])

        ax.plot_trisurf(t1, t2, R_vals, cmap="viridis", edgecolor="none")
        ax.set_xlabel(f"{combo[0]} Thickness (m)")
        ax.set_ylabel(f"{combo[1]} Thickness (m)")
        ax.set_zlabel("R (m²K/W)")
        ax.set_title(f"R-value Surface (Total Thickness = {total_thickness:.2f} m)")
        graph_file = out_path / f"R_surface_{'_'.join(combo)}.png"
        plt.savefig(graph_file)
        plt.close()
        print(f"Sweep complete. Results saved to:\n  - {txt_file}\n  - {graph_file}")
        return
    
    #for 3 materials or more plots 'iteration number'
    # This plot is now *much more useful* as it shows
    # R-value vs. combination #, for a fixed total thickness.
    else:
        y = [r[-1] for r in results]
        plt.plot(range(len(y)), y, 'o', alpha=0.5, markersize=4)
        plt.xlabel("Combination Number")
    
    #format and saving for all
    plt.ylabel("R (m²K/W)")
    plt.title(f"R-Value Sweep: {' + '.join(combo)} (Total Thickness = {total_thickness:.2f} m)")
    plt.grid(True)
    plt.tight_layout()

    graph_file = out_path / f"R_sweep_{'_'.join(combo)}.png"
    plt.savefig(graph_file)
    plt.close()

    print(f"Sweep complete. Results saved to:\n  - {txt_file}\n  - {graph_file}")

#------------------------------------------------------------------------------

"run directly plus example"

if __name__ == "__main__":
    #load material dictionary (requirement)
    data = load_materials()
    print("Loaded materials dictionary:\n", data, "\n")

    #example usage for compute R
    R_example = compute_R("Expanded Polystyrene (EPS)", 0.2, "mineral_wool", 0.05)
    print(f"total R = {R_example:.4f} m²K/W")
    
    sweep_fixed_total_thickness(
        combo=("Expanded Polystyrene (EPS)", "mineral_wool", "Polyurethane (PUR)"),
        total_thickness=0.4,
        step_size=0.02
    )