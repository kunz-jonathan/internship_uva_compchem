#!/bin/bash
#SBATCH --job-name=index_creation
#SBATCH --output=smaller_tpr_for_hydrogen_calc_%j.log     
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpuRTX-2nd


source /home/spack-user-2025/spack/share/spack/setup-env.sh


spack load gromacs@2024.3/qtc


cd jobs
for i in {1..103}
  do
    if [ -d "job_$i" ]
      then
        echo ================================================
        echo start job: $i
        
        cd job_$i

        echo 1 | gmx_mpi convert-tpr -s ./6.md_sim/md_0_1.tpr -o tpr_smaller_system.tpr -n ./index_mmpbsa.ndx 

        echo finished job: $i

        cd ../

    fi
done   



