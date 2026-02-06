#!/bin/bash
#SBATCH --job-name=align
#SBATCH --output=align_%j.log     
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpuRTX-4th,gpuRTX-3rd,gpuRTX-2nd


source /home/spack-user-2025/spack/share/spack/setup-env.sh

spack load gromacs@2024.3/qtc

echo $(spack find --format='{version}' gromacs | sort -V | tail -n 1)
echo $(gmx_mpi --version)

cd jobs

for i in {1..103}
  do
    if [ -d "job_$i" ]
      then
        echo ================================================
        echo start job: $i
        cd job_$i

        cd ./7.analysis

        echo 1 1 | gmx_mpi trjconv -f ../6.md_sim/md_0_1.xtc -s ../6.md_sim/md_0_1.tpr -o mdPBC_1 -pbc cluster
        echo 3 1 | gmx_mpi trjconv -f mdPBC_1.xtc -s ../5.equilibriation_second/npt.gro -o mdAlignCG_1 -fit rot+tran
        echo 1 | gmx_mpi trjconv -f mdAlignCG_1.xtc -s ../5.equilibriation_second/npt.gro -o mdAlignCG_1.pdb -dt 1000 

        echo finished job: $i
        cd ../..

    fi
done   
