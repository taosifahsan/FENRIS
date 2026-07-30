# this is a Doxygen directive used for documentation
## [QuasiLinear_2d python]

import numpy as np
import matplotlib.pyplot as plt
import asgard
import os
import argparse
import matplotlib as mpl
import subprocess
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

def available_cpus():
    """Return the scheduler/local CPU count used for OpenMP and parallel builds."""
    for key in ("SLURM_CPUS_PER_TASK", "SLURM_CPUS_ON_NODE", "PBS_NP", "NSLOTS"):
        val = os.environ.get(key)
        if val and val.isdigit() and int(val) > 0:
            return int(val)
    try:
        return len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        pass
    return os.cpu_count() or 1

def run_checked(command, *, env=None):
    print(" ".join(command))
    subprocess.run(command, check=True, env=env)
    
# This example is related to QuasiLinear_2d.cpp
# 1. This will run the C++ executable and generate an hdf5 file
#    - The file contains a snapshot of the PDE solution
# 2. The snapshot is loaded using the asgard python bindings
# 3. The solution and the exact solution are plotted together
output_filename = 'QL_2d_final.h5'
input_filename = 'input.txt'

if __name__ == '__main__':
    params = read_input_file(input_filename)
    # build folder and install folder names can be different
    if os.path.isfile('QuasiLinear_2d'):
        exefilename = 'QuasiLinear_2d'
    else:
        print("You must first build this project using CMake, e.g.,")
        print("  mkdir build")
        print("  cd build")
        print("  cmake ..")
        print("  make -j")
        print("")
        print("Then run this script from the build folder")
        exit(1)

    parser = argparse.ArgumentParser(description="Control ASGarD solver execution")
    parser.add_argument( "--only_plot", "-o",
        action="store_true",
        help="Do Not Run the ASGarD solver"
    )
    parser.add_argument( "--threads", "-j",
        type=int,
        default=None,
        help="OpenMP/build thread count. Defaults to OMP_NUM_THREADS, scheduler CPUs, or local CPUs."
    )
    args = parser.parse_args()
    
    if not args.only_plot or not os.path.exists(output_filename):
        threads = args.threads
        if threads is None:
            omp_env = os.environ.get("OMP_NUM_THREADS")
            threads = int(omp_env) if omp_env and omp_env.isdigit() else available_cpus()

        env = os.environ.copy()
        env.setdefault("OMP_NUM_THREADS", str(threads))
        print(f"OMP_NUM_THREADS={env['OMP_NUM_THREADS']}")

        run_checked(["make", "-j", str(max(1, threads))], env=env)

        launcher = os.path.join(os.getcwd(), "ql2d_omp_launch.sh")
        if os.path.exists(launcher):
            run_checked(["/usr/bin/env", "bash", launcher,
                         f"./{exefilename}", "-if", input_filename, "-of", output_filename],
                        env=env)
        else:
            run_checked([f"./{exefilename}", "-if", input_filename, "-of", output_filename],
                        env=env)

    print("asgard: running the quasi linear 2d")

    # the example above will run for 10 time steps and the -w 10 options
    # will tell the code to output on the final 10-th step
    
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

    f_min_abs = np.abs(f.min())
    print("Maximum negative value = " + str(f_min_abs))
    
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
    ax1.set_xlabel(r"$v_\parallel$", fontsize = 15)
    ax1.set_ylabel(r"$v_\perp$", fontsize = 15)
    
    ax2.set_xlabel(r"$v_\parallel$", fontsize = 15)
    ax2.set_ylabel(r"$v_\perp$", fontsize = 15)
    
    ax1.set_aspect('equal')
    ax2.set_aspect('equal')
    
    fig.colorbar(comp, ax=[ax1, ax2], shrink=0.5, label=r"$\log|f|$")
    
    fig_3d = plt.figure();
    ax_3d = fig_3d.add_subplot(projection = '3d')
    
    # plot the image of the computed values
    comp_3d = ax_3d.plot_surface(vpar, vperp, np.log(f)/np.log(10), cmap='jet')
    # set the colorbar
    ax_3d.set_xlabel(r"$v_\parallel$", fontsize = 15)
    ax_3d.set_ylabel(r"$v_\perp$", fontsize = 15)
    ax_3d.set_zlabel(r"$\log|f|$", fontsize = 15)
    fig.colorbar(comp_3d)
    
    # python plotting:
    # create a new figure with aspect ratio (14, 8)
    
    # Define a regular vpar (w) axis and vperp axis
    w  = np.linspace(-u_max, u_max, N)   # vpar range: u*cos(th) in [-10, 10]
    vp = np.linspace(0, u_max, N)     # vperp range: u*sin(th) in [0, 10]
    w_2d, vp_2d = np.meshgrid(w, vp, indexing='ij')

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
    ax_s.semilogy(w, F_w)
    ax_s.semilogy(w, -F_w)
    ax_s.semilogy(w,F_w_a)
    ax_s.set_xlabel(r'$w$')
    ax_s.set_ylabel(r'$F(w)$')
    ax_s.grid('True')
    plt.show()
    
    
