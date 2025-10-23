# Parameters
width_house = 9.0  # meters
length_house = 4.0 # meters
height_house = 3.0 # meters


# Calculations
vol_house = width_house * length_house * height_house  # m^3
density_air = 1.225  # kg/m^3
mass_air = vol_house * density_air  # kg
C = mass_air * 1005  # J/K

R = n


def simulate(R):
    # Define the ODE for a given R
    def dTdt(t, T):
        T_out = np.sin(t / (3600 * 12)) * 10 + 20  # one cycle every 24 hours
        return (T_out - T) / (R * C)