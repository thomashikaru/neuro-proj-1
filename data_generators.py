import numpy as np


def late_pulse():
    I_1_vals = np.full((100), 0.5)
    I_2_vals = np.full((100), 0.5)
    I_1_vals[70:80] = 0.8
    return I_1_vals, I_2_vals


def early_pulse():
    I_1_vals = np.full((100), 0.5)
    I_2_vals = np.full((100), 0.5)
    I_1_vals[20:30] = 0.8
    return I_1_vals, I_2_vals
