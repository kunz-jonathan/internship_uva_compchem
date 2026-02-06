#!/bin/bash
#SBATCH --job-name=index_creation
#SBATCH --output=index_creation_%j.log     
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpuRTX-2nd,gpuRTX-4th

source /home/spack-user-2025/spack/share/spack/setup-env.sh


spack load gromacs@2024.3/qtc


cd jobs

for i in {1..103}
  do
    if [ -d "job_$i" ]
      then
        echo item: $i
        cd job_$i

        gmx_mpi make_ndx -f ./5.equilibriation_second/npt.gro -o ./index_mmpbsa.ndx <<EOF
ri 1-160
ri 161-200 & ! r SOL & ! r NA & ! r CL
ri 1-200 & ! r SOL & ! r NA & ! r CL
17 & a CA
18 & a CA
name 17 onlyProtein
name 18 onlyPeptide 
name 19 Prot+Pept
name 20 CAonlyProtein
name 21 CAonlyPeptide
q
EOF


        cd ../

    fi
done   



