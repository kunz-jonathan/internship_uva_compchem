#!/bin/bash
#SBATCH --job-name=hbond
#SBATCH --output=hbond_donor_%j.log     
#SBATCH --nodes=1
#SBATCH --ntasks=20
#SBATCH --cpus-per-task=1
#SBATCH --partition=CP2K-ib

ulimit -S -c 0
ulimit -s unlimited
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

mpirun -n $SLURM_NTASKS python3 /home/kunzj/internship_uva_compchem/X.Scripts/md_sim_analysis/hydrogen_bond/mpi4py_scheduler.py