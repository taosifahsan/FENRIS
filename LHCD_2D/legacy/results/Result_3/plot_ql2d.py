# this is a Doxygen directive used for documentation
## [QuasiLinear_2d python]

import numpy as np
import matplotlib.pyplot as plt
import asgard
import os
import argparse
import matplotlib as mpl
from scipy.interpolate import griddata
from scipy.integrate import cumulative_trapezoid

def read_input_file(filename):
    """Reads key-value parameters from the input file."""
    params = {}
    with open(filename, "r") as f:
        for line in f:
            line = line.split('#', 1)[0]  # Remove comments
            if ':' in line:
                key, val = line.split(':', 1)
                params[key.strip()] = val.strip()
    return params
    
def D_rf(w, D, w_1,w_2):
    return D * (w_1<w) * (w<w_2);
    
# This example is related to QuasiLinear_2d.cpp
# 1. This will run the C++ executable and generate an hdf5 file
#    - The file contains a snapshot of the PDE solution
# 2. The snapshot is loaded using the asgard python bindings
# 3. The solution and the exact solution are plotted together
output_filename = 'QL_2d_final.h5'
input_filename = 'input.txt'
if __name__ == '__main__':
    params = read_input_file(input_filename)
    # using the ASGarD python module to read from the file
    snapshot = asgard.pde_snapshot(output_filename)

    print("creating 2d plot")

    N = int(params.get("num_points"))
    w_1 = float(params.get("cut_w_min"))
    w_2 = float(params.get("cut_w_max"))
    D = float(params.get("cut_u"))
    Zi = float(params.get("Zi"))
    
    u_max = float(params.get("u_max"))
    
    f, u, th = snapshot.plot_data2d(((), ()), num_points = N)

    log_f = np.log(np.abs(f));
    
    val_max = log_f.max();
    val_min = val_max - 12;

    mu = np.cos(th);
    vperp = u * np.sin(th)
    vpar = u * np.cos(th)
    
    # python plotting:
    # create a new figure with aspect ratio (14, 8)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))
    
    # plot the image of the computed values
    val_levels = np.linspace(val_min,val_max,50)
    comp = ax1.contour(vpar, vperp, np.log(f)/np.log(10),
                        levels=50, cmap="jet", vmin=val_min, vmax=val_max)
    comp_minus = ax2.contour(vpar, vperp, np.log(-f)/np.log(10),
                        levels=50, cmap="jet", vmin=val_min, vmax=val_max)
    # set the colorbar
    ax1.set_xlabel(r"$v_\parallel/v_e$", fontsize = 15)
    ax1.set_ylabel(r"$v_\perp/v_e$", fontsize = 15)
    
    ax2.set_xlabel(r"$v_\parallel/v_e$", fontsize = 15)
    ax2.set_ylabel(r"$v_\perp/v_e$", fontsize = 15)
    
    ax1.set_aspect('equal')
    ax2.set_aspect('equal')
    
    fig.colorbar(comp, ax=[ax1, ax2], shrink=0.5, label=r"$\log|f|$")
    
    fig_3d = plt.figure();
    ax_3d = fig_3d.add_subplot(projection = '3d')
    
    # plot the image of the computed values
    comp_3d = ax_3d.plot_surface(vpar, vperp, np.log(f)/np.log(10), cmap='jet')
    # set the colorbar
    ax_3d.set_xlabel(r"$v_\parallel/v_e$", fontsize = 15)
    ax_3d.set_ylabel(r"$v_\perp/v_e$", fontsize = 15)
    ax_3d.set_zlabel(r"$\log|f|$", fontsize = 15)
    fig.colorbar(comp_3d)
    
    # python plotting:
    # create a new figure with aspect ratio (14, 8)
    
    # Define a regular vpar (w) axis and vperp axis
    w  = np.linspace(-u_max, u_max, N)   # vpar range: u*cos(th) in [-10, 10]
    vp = np.linspace(0, u_max, N)     # vperp range: u*sin(th) in [0, 10]
    w_2d, vp_2d = np.meshgrid(w, vp, indexing='ij')

    E = w**2/2.0
    # Scatter (vpar, vperp, f) points and resample
    f_cart = griddata(
        points=(vpar.ravel(), vperp.ravel()),
        values=f.ravel(),
        xi=(w_2d, vp_2d),
        method='cubic',
        fill_value=0.0
    )
    # Integrate over vperp axis
    F_w = np.trapezoid(f_cart * 2 * np.pi * vp[np.newaxis, :], vp, axis=1)
    
    I = -w/(1.0 + 2.0/(2.0 + Zi) * D_rf(w,D,w_1,w_2) * w**3)
    integral = cumulative_trapezoid(I, w, initial = 0) - u_max**2/2.0
    
    C = 1/np.sqrt(2 * np.pi)
    
    F_w_a = C * np.exp(integral)
    
    fig_s = plt.figure()
    ax_s = fig_s.add_subplot()
    ax_s.semilogy(E, F_w)
    ax_s.semilogy(E, -F_w)
    ax_s.semilogy(E,F_w_a)
    ax_s.set_xlabel(r'$E=(v_\parallel/v_e)^2/2$')
    ax_s.set_ylabel(r'$F(E)$')
    ax_s.grid('True')
    plt.show()
    
    
