"""
Script to generate chemical profile for tr=0
"""
import numpy as np
from scipy.integrate import odeint
import trans_instab as ti
from scipy.optimize import brentq
import os, time
savepath = os.getcwd()
#=============================
# Discretization of space
N = 4e3  # Number of nodes in a uniform grid
H = 1  # Height of gel (given in normalized coordinate)
X2 = np.linspace(0, H, N) # Position of nodes

# Material properties
Nvb = 1e-2  # Stiffness value for the lower part of the gel
n = 5  # Stiffness ratio between uppper and lower layer
eta = 0.1  # Fraction of gel being the upper layer for layered structure
trans_frac = 0.002  # Transition fraction for layered structure
kappa = 10  # Parameter to define the shape of a continous stiffness gradient
stiff_grad = 'layer'  # Gradient of stiffness - continous or bilayered structure (i.e. 'cont' or 'layer')
chi = 0.5  # Chi parameter
D = 1e-9  # Diffusion coefficient

# Normalized chemical potentials
mu_init = -2  # Initial chemical potential through gel
mu_top = 0  # Chemical potential at top at t=0

# Discretization of time
tau = H**2/D  # Normalized time
end_time = 0.1  # Fraction of tau for last time step
steps = int(2e3)  # Number of time steps
tspan = np.linspace(0, end_time*tau, steps)  # Time steps
#=============================
## PDE solution
filename = r'chem_profile_tr0_Nv'+str("{:.0e}".format(Nvb))+'_n'+str(n).replace('.','c')+'_'+stiff_grad
print('Running ', filename)
if stiff_grad == 'cont':
    Nv = ti.cont_stiff(Nvb, n , kappa, X2)
if stiff_grad == 'layer':
    Nv = ti.layer_stiff(Nvb, n , eta, trans_frac, X2)

# Define intial conditions
l0 = np.zeros(len(Nv))
for i in range(len(Nv)):
    l0[i] = brentq(ti.l0_calc, 1.0005, 50, args=(Nv[i], chi, mu_init)) # Find inital swelling by s22 = 0
init = np.ones(X2.shape) # Initial stretch in body of gel
init[-1] = brentq(ti.l2_from_mu, 1.0005, 50, args=(mu_top, Nv[-1], chi, l0[-1])) # Boundary condition at surface

# Solve l2 profile in
t0 = time.time()
l2_profile = odeint(ti.dl2dt_func, init, tspan, args=(Nv, chi, l0, D, X2))
t1 = time.time()
print('Computing time for solving l2 in time ', t1-t0)

# Calculate chemical profile
mu_profile = ti.mu_from_l2(l2_profile, Nv, chi, l0)

# Save data
data = np.c_[tspan/tau, mu_profile]
np.savetxt(savepath+'/'+filename+'.txt', data, delimiter=',')
