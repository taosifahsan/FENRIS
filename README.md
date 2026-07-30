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

COMMENTS: 
The user can tweak run.sh to set how many CPUs will be used in different stages of
the code. The solver parallelization is best utilized by setting open_mp threads to 2. For pretabulation
or plot, speed is proportional to CPUs used.


