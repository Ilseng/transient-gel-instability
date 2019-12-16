# transient-gel-instability
Scripts for solving the stability of a hydrogel plate during transient swelling. The code is based on a linear perturbation analysis as described by Wu et al. Mechanics of Materials, 105 (2017), 38–147, http://dx.doi.org/10.1016/j.mechmat.2016.11.005, solving the eigenvalue problem with the finite difference method. Solves the stability problem in the fast and slow diffusion limits. 

## Usage
The code is written for Python 3.6 and requires numpy, scipy, and matplotlib. The main.py file runs the stability calculations code with chemical potential profiles as input, while trans_instab.py holds relevant functions. Chemical potential profiles through time must be generated for representative material properties from generate_profile_tr0.py and/or generate_profile_trinf.py.  
