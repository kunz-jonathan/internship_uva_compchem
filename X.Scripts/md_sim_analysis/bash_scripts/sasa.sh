#!/bin/bash
#SBATCH --job-name=sasa_ana
#SBATCH --output=sasa_%j.log     
#SBATCH --nodes=1
#SBATCH --ntasks=1

#SBATCH --partition=gpuRTX-4th


source /home/spack-user-2025/spack/share/spack/setup-env.sh
echo spack find --format='{version}' gromacs | sort -V | tail -n 1
echo spack load gromacs

spack load gromacs@2024.3/qtc

echo gmx_mpi --version



cd jobs


for i in {1..103}
  do
    if [ -d "job_$i" ]
      then
        echo ================================================
        echo start job: $i

        cd job_$i
        cd ./7.analysis

        gmx_mpi make_ndx -f ../5.equilibriation_second/npt.gro -o ./index.ndx <<EOF
ri 82-108 
name 17 onlyProtein
q
EOF


        echo 17 | gmx_mpi sasa -s ../5.equilibriation_second/npt.gro -f mdAlignCG_1.xtc -n ./index.ndx -o sasa_prot.xvg -xvg none -tu ns -b 5 -dt 1	
        
        cd ../..

        echo finished job: $i

    fi
done   


