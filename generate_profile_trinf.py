"""
Script to generate chemical profile for tr=infty
"""
import numpy as np
import os
savepath = os.getcwd()

# Normalized potential
mu_bot = -2
mu_top = -1e-20

# Timesteps - ueven spacing
x = np.linspace(0,1, int(1e4))
step = x**0.5

# Save chemical potential
mu = mu_bot*(1-step)+mu_top*step
data = np.c_[step, mu, mu]
np.savetxt(savepath+r'\chem_profile_trinf.txt', data, delimiter=',')
