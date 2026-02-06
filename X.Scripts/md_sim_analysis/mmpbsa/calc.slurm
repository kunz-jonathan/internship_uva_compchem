#!/bin/bash
#SBATCH --job-name=mmpbsa_job
#SBATCH --partition=EPYC-Milan
#SBATCH --ntasks=128
#SBATCH --output=milan_mmpbsa_%j.log

# Load Spack
source /home/spack-user-2025/spack/share/spack/setup-env.sh

ulimit -S -c 0
ulimit -s unlimited
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

source /home/kunzj/.bashrc
mamba activate gmxMMPBSA



# Arguments: working directory and ligand directory e.g. /home/kunzj/whole_pipeline/DATA/mmpbsa/jobs
WORKDIR=/home/kunzj/internship_uva_compchem/Data/iteration_0/md_sim_analysis/mmpbsa/jobs



# Loop over ligand directories
LIG_PATH_PARENT="$WORKDIR/"

for lig_idx_number in {1..103}; do
    lig_idx="$LIG_PATH_PARENT/job_$lig_idx_number"
    # Only process directories
    if [ -d "$lig_idx" ]; then
        pdb_file="$lig_idx/mdAlignCG_1.pdb"
        
        
        # Change to the ligand-specific working directory
        cd "$WORKDIR/$(basename "$lig_idx")" || continue



        echo start "$WORKDIR/$(basename "$lig_idx")"
        # Run gmx_MMPBSA
        mpirun -np $SLURM_NTASKS gmx_MMPBSA -O \
            -i mmpbsa.in \
            -cs "$pdb_file" \
            -ct mdAlignCG_1.xtc \
            -ci index_mmpbsa.ndx \
            -cg 17 18 \
            -cp topol.top \
            -o test_results_energy.dat \
            -eo test_results_energy.csv \
            >& mmgbsa.log
        echo finished "$WORKDIR/$(basename "$lig_idx")"
    fi
done
