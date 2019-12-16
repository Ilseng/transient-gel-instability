"""
Script to solve linear perturbation analysis for a hydrogel with gradient
stiffness under transient swelling using a finite difference approach.
Written for Python 3.6.1

Written by: Arne Ilseng
Department of Structural Engineering,
Norwegian University of Science and Technology

The perturbation analysis is based on the work by Wu et al. published in Mechanics of Materials in 2017
doi: http://dx.doi.org/10.1016/j.mechmat.2016.11.005
"""
import numpy as np
import matplotlib.pyplot as plt
import os, time
import pandas as pd
import trans_instab as ti
path = os.getcwd()
# ==============================================================================
# --- Initialize
# ==============================================================================
#  Input parameters
num_data = int(20)  # Number of wavenumbers in first calculation
wH_low = 1  # Lowest wavenumber
wH_heigh = 40  # Heighest wavenumber
wH_list = np.linspace(wH_low, wH_heigh, num=num_data)  # Wavenumber discretization
Nvb = 1e-2  # Stiffness value for the lower part of the gel
chi_b = 0.5  # Chi parameter of gel
n = 5  # Stiffness ratio between uppper and lower layer
eta = 0.1  # Fraction of gel being an upper layer if layer structure
stiff_grad = 'layer'  # Stiffness through gel, continous or bilayer structure (i.e. 'cont' or 'layer')
trans_frac = 0.002  # Transition fraction for bilayer structure
kappa = 10  # Shape of stiffness gradient for continous change

#  Grid parameters
mstar = 5e3  #  Number of nodes over gel with even spacing
alpha = 1e-2  #  Node distance increase for ueven part of grid
rho = 0.12  #  Upper fraction of gel with even node spacing

#  Normalized ramping time for boundary condition
tr = 'inf' # Relative ramping time tr for changing boundary condition (i.e. '0' or 'inf')

#  Set filename
filename = ('Nv'+str("{:.0e}".format(Nvb))+'_n'+str(n).replace('.','c')+
'_node'+str(int(mstar))+'_rho'+str(rho).replace('.','c')+
'_alpha'+str(alpha).replace('.','c')+'_tr'+tr+'_'+stiff_grad)
print('Working on ' + filename)

#  Read chemical profile (first colum being time, following columns being node values)
t0 = time.time()
zero_profile = (path+'/chem_profile_tr0_Nv'+str("{:.0e}".format(Nvb))+'_n'
    +str(n).replace('.','c')+'_layer.txt') # Profile data for tr=0
inf_profile = path+'/chem_profile_trinf.txt' # Profile data for tr=inf
if tr=='inf':
    try:
        chem_profile_data =  np.array(pd.read_csv(inf_profile, delimiter = ',', header=None))
    except:
        print('Chemical profile was not found for tr='+tr)
        print('Generate profile with "generate_profile_trinf.py"')
        exit()
elif tr=='0':
    try:
        chem_profile_data =  np.array(pd.read_csv(zero_profile, delimiter = ',', header=None))
    except:
        print('Chemical profile was not found for tr='+tr)
        print('''Generate profile by running "generate_profile_tr0.py" with appropriate material parameters''')
        exit()
else:
    print ('Normalized ramping time "' +tr +'" not known')
    print ('Use tr="0" or tr="inf"')
    exit()
chem_profile = chem_profile_data[:,1:]  # Keep only node values
time_steps = len(chem_profile[:,0])  # Number of time steps in the chemical profile data
step_low = 1  # Assume that instability has not happend in the initial state
step_heigh = time_steps-1
t1 = time.time()
print('Reading the chemical profile took', t1-t0, 'seconds')

#  Define finite difference grid
X2, dx, dn = ti.grid(mstar, alpha, rho)

#  Define material parameters for each node
if stiff_grad == 'cont':
    Nv = ti.cont_stiff(Nvb, n, kappa, X2)
if stiff_grad == 'layer':
    Nv = ti.layer_stiff(Nvb, n , eta, trans_frac, X2)
chi = chi_b*np.ones_like(X2) # Set same chi value for whole gel

# ==========================================================================
# --- Run code, save and plot obtained results
# ==========================================================================
#  Initialize result arrays and loop variables
print('Solving ....')
t0 = time.time()
wH_res, lam_res, step_res = ti.solve(X2, dx, dn, Nv, chi, chem_profile,
    wH_list, step_low, step_heigh)
t1 = time.time()
print('The computation took', t1-t0, 'seconds')

#  Save results
np.savetxt(path+'/'+filename+'.txt', [wH_res, lam_res, step_res], delimiter=';')

#  Plot results
plt.figure()
plt.plot(2*np.pi/np.array(wH_res), lam_res, '-o')
plt.xlabel(r'Normalized wavelength $\bar{\Lambda}$ [-]')
plt.ylabel(r'Critical swelling ratio $\lambda_c$ [-]')
plt.show()
