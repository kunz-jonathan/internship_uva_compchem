#!/bin/bash

mkdir -p npt_run md_run

cd ./npt_run

# Short equilibration
# gmx_mpi grompp -f ../../../npt_frame.mdp -c ../../../config_files/confXXX.gro -r ../../../config_files/confXXX.gro -p ../../../../1.topology/topol.top -n ../../../../6.pull/index.ndx -o nptXXX.tpr -maxwarn 1 
# gmx_mpi mdrun -deffnm nptXXX -v


### pul-sim ###

cd ../md_run


# Umbrella run
gmx_mpi grompp -f ../../../md_frame.mdp -c ../../../config_files/confXXX.gro -r ../../../config_files/confXXX.gro -p ../../../../1.topology/topol.top -n ../../../../6.pull/index.ndx -o umbrellaXXX.tpr
gmx_mpi mdrun -deffnm umbrellaXXX -nb gpu >& ./pull.out

cd ../
