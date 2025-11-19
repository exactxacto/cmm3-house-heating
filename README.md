
## (i) Descriptions

### 1. thermal_resistance/src/main.py

* Reads material thermal conductivities.
* Generates all combinations of insulation layers and thicknesses.
* Computes total thermal resistance (R-value) for each configuration.
* Outputs results to: thermal_resistance/out/R_results_Expanded Polystyrene(EPS)_mineral_wool_Polyurethane (PUR).txt

### 2. MIDAS/interpolateV3.py

* Loads outdoor temperature data (e.g. MIDAS / Met Office).
* Cleans and interpolates missing data.
* Removes long-term trends.
* Performs Fourier analysis to reconstruct a smooth yearly temperature cycle.

### 3. src/v2_simulate_solar_irradiance.py

* Loads R-values from the thermal resistance output, solar irradiance measurements from SolRad.xlsx. and ambient temperature.
* Converts solar irradiance into an equivalent heat-gain term.
* Produces an effective outdoor temperature for each wall configuration.
* Outputs: T_out_df (hourly effective outdoor temperature for every R-value and R_edd_dict (mapping of configuration labels to R-values).

### 4. src/solve_ODE.py

* Loads T_out_df and R_eff_dict.
* Defines a lumped-capacity indoor temperature model.
* Solves first-order ODE for indoor temperature over one year for each R-value.
* Saves results i.e. hourly indoor temperature for every wall configuration

### 5. comfort_cost_analysis/cost_combination.py

* Assigns material and thickness costs to each wall design.
* Calculates total insulation cost for each R-value configuration.
* Outputs a combined cost table (combinations_and_costs.csv)

### 6. comfort_cost_analysis/main_analysis.py

* Loads indoor temperature simulations and cost data.
* Computes comfort metrics (e.g. % hours within 18–24 °C).
* Builds curves for comfort vs R-value and cost vs R-value.
* Performs search/root-finding to determine the R-value that meets a target comfort level at minimum cost.
* Generates plots showing the optimal design

### 7. easteregg.py

* Inspiration and soundtrack with no impact on numerical workflow.

## (ii) Order

To execute the project directory, the above scripts must be run in that sequential order. For easy the order is also listed below:

1. thermal_resistance/src/main.py
2. MIDAS/interpolateV3
4. src/v2_simulate_solar_irradiance
5. src/solve_ODE
6. comfort_cost_analysis/cost_combination
7. comfort_cost_analysis/main_analysis
8. easteregg.py

## (iii) Numerical Methods Locations

### 1. Interpolation/Regression

* MIDAS/interpolateV3.py : interpolates missing temperature data and applies Fourier reconstruction (sinusoidal regression)
* v2_simulate_solar_irradiance.py : aligns and interpoolates solar/temperature time series
* comfort_cost_analysis/main_analysis.py : interpolates comfort vs R-value and cost vs R-value to produce smooth curves.

### 2. ODE Solving

* src/solve_ODE.py : Uses scipy.integrate.solve_ivp to integrate the indoor temperature ODE

### 3. Root Finding / Optimisation

* comfort_cost_analysis/main_analysis.py : solves for the R-value where comfort meets a target value and chooses the minimum-cost configuration satisfying comfort constraints.


