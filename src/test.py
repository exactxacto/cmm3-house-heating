import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

from v2_simulate_solar_irradiance import T_out_df, R_eff_dict, time_hours, n_hours

try:
    T_out_df  # defined earlier
except NameError:
    raise RuntimeError("T_out_df not found — run the generation code first.")

columns = list(T_out_df.columns)
n_hours = len(T_out_df.index)
time_hours = np.asarray(T_out_df.index)           # 0..n_hours-1

# Extract R_eff mapping (dictionary keyed by column names)
R_eff_dict = T_out_df.attrs.get('R_eff', None)
if R_eff_dict is None:
    try:
        R_eff  # numpy array from previous code
        R_eff_dict = dict(zip(columns, R_eff))
    except NameError:
        raise RuntimeError("R_eff metadata not found. Put R_eff into T_out_df.attrs['R_eff'] or define R_eff array.")

def simulate(R_label):
    T_out_series = np.ravel(T_out_df[R_label].to_numpy().astype(float))
    R_eff = float(R_eff_dict[R_label])
    print(T_out_df)
simulate(R_label)
