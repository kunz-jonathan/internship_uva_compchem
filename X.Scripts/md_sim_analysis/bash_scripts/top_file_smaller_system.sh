#!/bin/bash
#SBATCH --job-name=index_creation
#SBATCH --output=smaller_top_%j.log     
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
        echo item: $i
        cd job_$i

        echo 1 | gmx_mpi editconf -f ./5.equilibriation_second/npt.gro -o top_smaller_system.gro -n ./index_mmpbsa.ndx 


        cd ../

    fi
done   



