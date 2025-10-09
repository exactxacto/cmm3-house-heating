from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt

T_sep1 = 25 # Average temperature on Sept 1

def temp_inside(T_out,T_in,R,C):
    dTdt = (T_out-T_in)/(R*C)
    print (dTdt)

temp_inside(30,15,0.03,0.1)
