To run the project properly, a number of scripts must be run in sequential order. The order to run them in is as follows:
1. /thermal_resistance/src/main.py
2. /MIDAS/interpolateV3
4. /src/v2_simulate_solar_irradiance
5. /src/solve_ODE
6. /comfort_cost_analysis/cost_combination
7. /comfort_cost_analysis/main_analysis
8. easteregg.py

MIDAS

Comfort Cost Analysis

References

v2_simulate_solar_irradiance: Takes the generated ambient temperature Fourier equation, solar heat flux values and outputs a Pandas DataFrame with effective outdoor temperature values and 
an appended list of R-values.

solve_ODE: Implements the actual ODE solver and outputs an Excel file with the internal temperature values for 231 R-value combinations. 

ReadME


SolRad - list of solar irradiance and solar heat flux values. 



(ii)

(iii) Where the three required numerical methods are located:

