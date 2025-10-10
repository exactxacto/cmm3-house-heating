# -*- coding: utf-8 -*-
"""
Created on Thu Oct  9 12:45:31 2025

@author: Cailean
"""
#-------------------------------------------------------------------

from pathlib import Path
import matplotlib.pyplot as plt
import itertools
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

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

"Sweeps combinations of 2 or 3 materials with varying thicknesses"
"combo: material names (1-3)"
"ranges: list of (start, end) thickness, one per material"
"steps: number of increments per range"
"out_folder: folder for output files"
def sweep_combinations(combo, ranges, steps=10, out_folder="out"):
   
    if "MATERIALS" not in globals():
        raise RuntimeError("MATERIALS not loaded. Run load_materials() first.")

    #ensure a thickness for each material
    if len(combo) != len(ranges):
        raise ValueError("Number of materials must match number of thickness ranges")

    #outfolder is in root folder
    out_path = repo_root() / out_folder
    out_path.mkdir(parents=True, exist_ok=True)

    #create sweep points for each material
    #make an evenly spaced list of values based on range and step
    sweep_points = [
        [r[0] + i * (r[1] - r[0]) / (steps - 1) for i in range(steps)]
        for r in ranges
    ]

    #create all combinations using itertools.product
    all_combos = list(itertools.product(*sweep_points))

    #empty dictionary for results
    results = []

    #loops through all combinations and computes sum of R based on compute_R function
    for thicknesses in all_combos:
        #build argument list for compute_R
        args = []
        for m, t in zip(combo, thicknesses):
            args.extend([m, t])

        R_total = compute_R(*args)
        results.append((*thicknesses, R_total))

    #save numerical results to txt for comparision/validation
    #creates name based on combination of results
    txt_file = out_path / f"R_combo_{'_'.join(combo)}.txt"
    with open(txt_file, "w") as f:
        header = "\t".join([f"{m}_thickness(m)" for m in combo]) + "\tR(m²K/W)\n"
        f.write(header)
        #creates tab seperated txt file with results
        for r in results:
            line = "\t".join([f"{x:.3f}" for x in r]) + "\n"
            f.write(line)

    #plot graphs
    plt.figure(figsize=(8, 5))

    #if 1 material sweep create 2d Line plot
    if len(combo) == 1:
        # Single material sweep
        x = [r[0] for r in results]
        y = [r[1] for r in results]
        plt.plot(x, y, marker="o")
        plt.xlabel(f"{combo[0]} Thickness (m)")
    
    #if 2 materials create 3d graph
    elif len(combo) == 2:

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")

        t1 = np.array([r[0] for r in results])
        t2 = np.array([r[1] for r in results])
        R_vals = np.array([r[2] for r in results])

        ax.plot_trisurf(t1, t2, R_vals, cmap="viridis", edgecolor="none")
        ax.set_xlabel(f"{combo[0]} Thickness (m)")
        ax.set_ylabel(f"{combo[1]} Thickness (m)")
        ax.set_zlabel("R (m²K/W)")
        ax.set_title(f"R-value Surface: {combo[0]} + {combo[1]}")
        graph_file = out_path / f"R_surface_{'_'.join(combo)}.png"
        plt.savefig(graph_file)
        plt.close()
        print(f"Sweep complete. Results saved to:\n  - {txt_file}\n  - {graph_file}")
        return
    
    #for 3 materials or more plots 'iteration number'
    else:
        y = [r[-1] for r in results]
        plt.plot(range(len(y)), y)
        plt.xlabel("Iteration")
    
    #format and saving for all
    plt.ylabel("R (m²K/W)")
    plt.title(f"R Sweep: {' + '.join(combo)}")
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
    
    #example 3 material sweep (Iteration graph and results txt)
    sweep_combinations(
        combo=("Expanded Polystyrene (EPS)", "mineral_wool", "Polyurethane (PUR)"),
        ranges=[(0.05, 0.2), (0.05, 0.1), (0.05, 0.1)],  # PUR varies 0.05–0.2, XPS varies 0.05–0.1
        steps=10
    )
    
    #example 2 material sweep (3d graph and results txt)
    sweep_combinations(
        combo=("Expanded Polystyrene (EPS)", "mineral_wool"),
        ranges=[(0.05, 0.2), (0.05, 0.1)],  # PUR varies 0.05–0.2, XPS varies 0.05–0.1
        steps=15
    )

