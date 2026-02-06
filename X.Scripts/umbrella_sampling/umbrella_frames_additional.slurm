#!/bin/bash
#SBATCH --job-name=umbrella_sampling
#SBATCH --output=md_%j.log     
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpuRTX-4th
#SBATCH --nodelist=gn3,gn2,gn1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=32


source /home/spack-user-2025/spack/share/spack/setup-env.sh
echo spack find --format='{version}' gromacs | sort -V | tail -n 1
echo spack load gromacs

spack load gromacs@$(spack find --format='{version}' gromacs | sort -V | tail -n 1) 
source ~/.bashrc


echo gmx_mpi --version

echo "SLURM_JOBID=$SLURM_JOBID"
echo "SLURM_JOB_NODELIST=$SLURM_JOB_NODELIST"
echo "SLURM_NNODES=$SLURM_NNODES "
echo "working directory= $SLURM_SUBMIT_DIR "
echo "tmp directory=$TMPDIR"
echo "NPROCS=$SLURM_NPROCS"
echo "Number of CPUs per task=$SLURM_CPUS_PER_TASK"
echo "Count of processors avaiable to the job on this node=$SLURM_JOB_CPUS_PER_NODE"
echo "Number of CPUs requested per allocated GPU=$SLURM_CPUS_PER_GPU"
echo "Number of GPUs requested=$SLURM_GPUS"
echo "Number of tasks=$SLURM_NTASKS"


export GMXLIB="/home/kunzj/internship_uva_compchem/X.Scripts/umbrella_sampling"

# no coredumps
ulimit -S -c 0
ulimit -s unlimited
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# identifier for scripts
script_pattern="*_run_frame.sh"

# loop over matching directories

for dir in frame-70 frame-60 frame-95 frame-65; do
    if [[ -d "$dir" ]]; then
        echo "Entering directory: $dir"
        cd "$dir" || continue
        for script in $script_pattern; do

            echo "Running $script inside $dir"
            bash "$script"
        done
        cd ..
    fi
done