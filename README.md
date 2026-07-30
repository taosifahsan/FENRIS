PROJECT DESCRIPTION:
This repository uses ASGarD, the adaptive sparse grid discretization library
(https://github.com/project-asgard/asgard) to explore various avenues of kinetic
theory-based Fokker-Planck equations for fusion plasma. Namely, 

1. LHCD_1D solves lower-hybrid current drive for electrons in one dimension
2. LHCD_2D solves lower-hybrid current drive for electrons in two dimensions
3. ICRF_1D solves ion-cyclotron radio frequency minority heating in one dimension
4. ICRF_2D solves ion-cyclotron radio frequency minority heating in two dimensions
5. plot_common is there for shared plot/animation methods used across files.

RUNNING:
Instructions to run the programs are straightforward:
    cd [folder]
    tools/run.sh  

PLOTTING:
run.sh already automatically takes care of the plotting. Instructions to plot individual
graphs or movies are also straightforward:
    cd [folder]
    plot/*.py  

PRETABULATION, ICRF_2D:
For ICRF_2D, the bounce averaging dimension reduction technique requires pretabulation of
coefficients to reduce runtime. Dependencies and staleness are set accordingly.

SLURM:
There is a submit.sh written in case users want to submit the code to Slurm. It wraps around
run.sh and submits it for the slurm job. 

COMMENTS: 
The user can tweak run.sh to set how many CPUs will be used in different stages of
the code. The solver parallelization is best utilized by setting open_mp threads to 2.
For pretabulation or plotting, speed is proportional to the number of CPUs used. When
submitting to slurm, editing run.sh is sufficient when distributing thread count because
submit.sh is a thin wrapper on run.sh.


