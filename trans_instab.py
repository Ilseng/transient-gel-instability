"""
File containing functions used in the stability analysis
Numba is used to compile functions with numerous calls
"""
import numpy as np
from scipy.optimize import brentq
from numba import jit

def grid(mstar, alpha, rho):
    """Define finite difference grid
    mstar: gives the density of nodes for X2>(1-rho)
    alpha: parameter to give the increase in spacing for X2<(1-rho)
    rho: upper fraction of grid with even spacing
    """
    delta = 1/(mstar-1)
    a = 0
    X2 = [1]
    while X2[-1]>0:
        delta = delta*(1+a)
        X2.append(X2[-1]-delta)
        if X2[-1]<(1-rho):
            a = alpha
    if abs(X2[-1])>abs(X2[-2]):
        X2 = X2[:-1]
    X2[-1] = 0
    X2 = np.flip(X2, axis=0)  # Make X2 go from 0 to 1
    dx = np.diff(X2)  # Distance between nodes, length of len(X2)-1
    hm = np.concatenate(([0], dx))  # Distance to next node going down
    hp = np.concatenate((dx, [0]))  # Distance to next node going up
    dn = 0.5*(hm+hp)  # Distance covered by each node, length of len(X2)
    return X2, dx, dn

@jit()
def mu_vec(X, step, chem_profile):
    """ Interpolation function to define mu value in node of coordinate X -
    assumes even spacing for chemical profile nodes """
    mu_vector = np.interp(X, np.linspace(0,1,len(chem_profile[0,:])),
                          chem_profile[int(np.ceil(step)),:])
    return mu_vector

@jit(nopython=True)
def l2_from_mu(l2, mu, Nv, chi, l0):
    """ Function to minimize to obtain the stretch l2 """
    J = l0**3*l2
    val = mu - np.log(1-1/J) - 1/J - chi/J**2 - Nv*(l2/l0-1/J)
    return val

def mu_from_l2(l2, Nv, chi, l0):
    """ Function to calculate mu based on the stretch l2 """
    J = l0**3*l2
    mu = Nv*l2/l0 + (np.log((J-1)/J)+(1-Nv)/J+chi/J**2)
    return mu

def l0_calc(l0_test, Nv, chi, mu):
    """ Function to minimize to obtain the initial stretch l0 """
    J = l0_test**3
    s22 = Nv/l0_test+np.log((J-1)/J)+(1-Nv)/J+chi/J**2-mu
    return s22

def dl2dt_temp_func(l2, Nv, chi, l0):
    "Function to calculate expression for dl2dt calculation"
    temp = (l0/l2-1/(l0**2*l2**2))*(Nv/l0+(1/((l0**3*l2-1)*l2)+(Nv-1)/(l0**3*l2**2)-2*chi/(l0**6*l2**3)))
    return temp

@jit()
def dl2dt_func(l2, t, Nv, chi, l0, D, X2):
    " Ode function to generate chemical potential - only written for even mesh"
    m = len(X2)
    dx = np.abs(X2[1]-X2[0]) # Assumes even node distribution
    dl2dt = np.zeros(m)
    # Define dudt for the internal nodes
    f1 = dl2dt_temp_func(l2[m-1:1:-1],Nv[m-1:1:-1], chi, l0[m-1:1:-1])*(l2[m-1:1:-1]-l2[m-2:0:-1])/dx
    f2 = dl2dt_temp_func(l2[m-2:0:-1], Nv[m-2:0:-1], chi, l0[m-2:0:-1])*(l2[m-2:0:-1]-l2[m-3::-1])/dx
    dl2dt[m-2:0:-1] = D/l0[m-2:0:-1]**3 * (f1 - f2)/dx
    # Boundary conditions
    dl2dt[-1] = 0 # No change with time at the surface (X2 = 1)
    dl2dt[0] = dl2dt[1] # Differential of l2 with respect to space is 0 at the bottom (X2 = 0)
    return dl2dt

@jit()
def calc_AB(l2, Nv, X2, xi, hm, hp, wH):
    """ Calculate A and B vectors to be put in M """
    # Calculate temporary variables
    dl2_dx = np.gradient(l2,X2, edge_order=2)
    f1 = (l2/Nv)*np.gradient(Nv*l2,X2, edge_order=2)
    f2 = (l2/Nv)*np.gradient(Nv*(xi-l2),X2, edge_order=2)
    f3 = (l2/Nv)*np.gradient(Nv*(xi+l2),X2, edge_order=2)
    a1, a2, b1, b2, g1, g2 = abg(hm, hp)
    temp1 = l2**(-2)*(f1-l2*dl2_dx)
    temp2 = l2**(-2)*(f3-(xi+l2)*dl2_dx)
    temp3 = (1/l2)*(xi+l2)
    temp4 = wH*xi
    # Calculate A and B matrices
    A = np.zeros((6, l2.shape[0]))
    B = np.zeros((6, l2.shape[0]))
    A[0,:] = a2+a1*temp1
    A[1,:] = -a1*temp4
    A[2,:] = b2+b1*temp1-wH**2*(1+l2*xi)
    A[3,:] = -b1*temp4-wH*(1/l2)*f1
    A[4,:] = g2+g1*temp1
    A[5,:] = -g1*temp4

    B[0,:] = a1*temp4
    B[1,:] = a2*temp3+a1*temp2
    B[2,:] = b1*temp4+wH*(1/l2)*f2
    B[3,:] = b2*temp3+b1*temp2-wH**2
    B[4,:] = g1*temp4
    B[5,:] = g2*temp3+g1*temp2
    return (A, B)

@jit(nopython=True)
def abg(hm, hp):
    """ Calculate alpha, beta and gamma for first and second derivative approximation """
    a1 = -hp/(hm*(hm+hp))
    g1 = hm/(hp*(hm+hp))
    b1 = -a1-g1
    a2 = 2/(hm*(hm+hp))
    g2 = 2/(hp*(hm+hp))
    b2 = -a2-g2
    return a1, a2, b1, b2, g1, g2

@jit()
def detM(step, X2, wH, nodes, Nv, chi, dx, chem_profile):
    """ Construct the total matrix M """
    #  Get stretch vector from mu vector
    mu_vector = mu_vec(X2, step, chem_profile)
    l2 = np.zeros(int(nodes))
    for i in range(len(l2)):
        l2[i] = brentq(l2_from_mu, 1.01, 20, args=(mu_vector[i],Nv[i], chi[i], 1.0))
    #  Define hm and hp with "ghost point" for hp
    hm = np.concatenate(([0], dx))
    hp = np.concatenate((dx, [dx[-1]]))
    #  Calculate xi
    xi = 1/l2 + (1/Nv)*(1/(l2-1)-(1/l2)-(2*chi)/(l2**2))
    #  Calculate coefficients for M
    A, B = calc_AB(l2[1:], Nv[1:], X2[1:], xi[1:], hm[1:], hp[1:], wH)

    #  Initialize M
    M = np.zeros((int(2*(nodes-1)), int(2*(nodes-1))))

    #  Define M for node 2 (bottom node being numbered as 1)
    M[0, 0:4] = A[2:,0]
    M[1, 0:4] = B[2:,0]

    #  Define M for node 3 to m-1
    for i in range(int(nodes-3)):
        M[2*i+2, 2*i:2*i+6] = A[:,i]
        M[2*i+3, 2*i:2*i+6] = B[:,i]

    #  Define last row of M (with use of ghost point)
    odx = 1/dx[-1]
    C1 = 2/odx*wH*l2[-1]
    C2 = -2/odx*wH*l2[-1]*(xi[-1]-l2[-1])/(xi[-1]+l2[-1])
    A_last = [A[0,-1]+A[4,-1], A[1,-1]+A[5,-1], A[2,-1]+C2*A[5,-1], A[3,-1]+C1*A[4,-1]]
    B_last = [B[0,-1]+B[4,-1], B[1,-1]+B[5,-1], B[2,-1]+C2*B[5,-1], B[3,-1]+C1*B[4,-1]]
    M[-2, -4:] = A_last
    M[-1, -4:] = B_last

    logdet = np.linalg.slogdet(M)
    return logdet[0]*np.exp(logdet[1])

def cont_stiff(Nvb, n , kappa, X2):
    "Function for continous stiffness gradient"
    Nv = Nvb + Nvb*(n-1)*(np.exp(kappa*X2)-1)/(np.exp(kappa)-1)
    return Nv

def layer_stiff(Nvb, n , eta, trans_frac, X2):
    "Function for bilayered stiffness with smooth transition"
    num_nodes = len(X2)
    node_sub = int(np.argmax(X2>(1-eta-trans_frac)))  # Number of nodes in bootom layer
    node_top = int(num_nodes-np.argmax(X2>(1-eta+trans_frac)))  # Number of nodes in upper layer
    if trans_frac > 0.0:
        node_trans = int(num_nodes-node_top-node_sub)
        x_trans = X2[node_sub:node_sub+node_trans]
        x_trans_norm = (x_trans-x_trans[0])/(x_trans[-1]-x_trans[0])
        #  Use smooth step from Nvb to Nvt
        s = 6*x_trans_norm**5-15*x_trans_norm**4+10*x_trans_norm**3
        Nv_trans = Nvb+Nvb*(n-1)*s
        Nv = (np.concatenate((Nvb*np.ones(node_sub),
            Nv_trans, n*Nvb*np.ones(node_top)), axis = None))
    else:
        Nv = (np.concatenate((Nvb*np.ones(node_sub),
            n*Nvb*np.ones(node_top)), axis = None))
    return Nv

def solve(X2, dx, dn, Nv, chi, chem_profile, wH_list, step_low, step_heigh):
    "Function to solve instability problem"
    #  Initialize result arrays
    step_res = []; wH_res = []; wH_run = [] # Empty arrays
    wH_list_temp = wH_list  # Set list of wH for calculation
    diff = 1; num_runs = 0  # Set loop variables
    num_nodes = len(X2)  # Number of nodes in grid

    ##  Find time step in mu profile file that gives instability
    while abs(diff)>0.05 and num_runs < 10: # Loop to refine results around min point
        for m, wH in enumerate(wH_list_temp):
            a = detM(step_low, X2, wH, num_nodes, Nv, chi, dx, chem_profile)
            b = detM(step_heigh, X2, wH, num_nodes, Nv, chi, dx, chem_profile)
            if np.sign(a) != np.sign(b):
                step = brentq(detM, step_low, step_heigh,
                    args=(X2, wH, num_nodes, Nv, chi, dx, chem_profile,), maxiter = 500)
                step_res.append(step)  # Append result to step_res array
                wH_res.append(wH)  # Append corresponding wave number
        wH_run += list(wH_list_temp)  # Update the list of wH values that have been run
        if len(step_res)>3:  # Improve list locally if more than 3 points of instability are found
            wH_res, step_res = (list(t) for t in zip(*sorted(zip(wH_res, step_res))))  # Sort the new result lists after wH value
            min_step_idx = np.argmin(step_res)  # Find index of first step with instability
            if min_step_idx == len(wH_res)-1:  # Minimum point at max wH value, add larger value
                wH_list_temp = np.linspace(wH_res[min_step_idx-1],
                    wH_res[min_step_idx]+np.diff(wH_res)[-1], 5)
            elif min_step_idx == 0:  # Minimum point at min wH value, add smaller valuex
                wH_list_temp = np.linspace(wH_res[min_step_idx]-np.diff(wH_res)[0],
                    wH_res[min_step_idx+1], 5)
            else:  # Minimum point in list, refine around min point
                wH_list_temp = np.linspace(wH_res[min_step_idx-1], wH_res[min_step_idx+1], 5)
        else:
            wH_list_temp = np.linspace(0.5, 50, len(wH_list)*2*(num_runs+1))
        diff = np.min(np.diff(wH_list_temp))  # Calculate diff for the last wH list
        wH_list_temp = [i for i in wH_list_temp if i not in wH_run]  # Avoid running wH values twice
        num_runs += 1

    ##  Sort results and calculate corresponding global stretch
    if len(step_res) > 2:
        # Sort results and initialize stretch results
        wH_res, step_res = zip(*sorted(zip(wH_res, step_res)))
        lamb_node = np.zeros(num_nodes) # For storing stretch values for each node
        lam_res = np.zeros(len(step_res)) # For global stretch of each time step
        # Calculate global stretch at instability
        for i, step_val in enumerate(step_res):
            mu = mu_vec(X2, step_val, chem_profile)
            for k in range(num_nodes):
                  lamb_node[k] = brentq(l2_from_mu, 1.0001, 50, args=(mu[k],Nv[k], chi[k], 1))
            lam_res[i] = np.sum(lamb_node*dn)
    else:
        print("Too few data points were found for ", filename)
        exit()
    return wH_res, lam_res, step_res

if __name__ == "__main__":
    # execute only if run as a script
    main()
