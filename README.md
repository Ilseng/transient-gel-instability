# transient-gel-instability
Scripts for solving the stability of a hydrogel plate with stiffness gradients during transient swelling. The code is based on a linear perturbation analysis and solves the eigenvalue problem in the fast and slow diffusion limits using the finite difference method. 

## Usage
The code is written for Python 3.6 and requires NumPy, SciPy, pandas, and Matplotlib. The main.py file runs the stability calculations code with chemical potential profiles as input, while trans_instab.py holds relevant functions. Chemical potential profiles through time can be generated for representative material properties using generate_profile_tr0.py and/or generate_profile_trinf.py.  
