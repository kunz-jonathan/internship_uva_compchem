#!/bin/bash
#SBATCH --job-name=rmsf_ana
#SBATCH --output=rmsf_%j.log     
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpuRTX-4th


source /home/spack-user-2025/spack/share/spack/setup-env.sh
echo spack find --format='{version}' gromacs | sort -V | tail -n 1
echo spack load gromacs

spack load gromacs@$(spack find --format='{version}' gromacs | sort -V | tail -n 1) 

echo gmx_mpi --version

cd jobs
for i in {1..103}
  do
    if [ -d "job_$i" ]
      then
        echo ================================================
        echo start job: $i
        
        cd job_$i
        
        gmx_mpi make_ndx -f ./5.equilibriation_second/npt.gro -o ./index.ndx <<EOF
ri 1-160 
ri 161-180 & ! r SOL & ! r NA & ! r CL
ri 1-160 & a CA
name 17 onlyProtein
name 18 onlyPeptide 
name 19 onlyProteinCA
q
EOF

        cd ./7.analysis

        echo 18 | gmx_mpi rmsf -s ../5.equilibriation_second/npt.gro -f mdAlignCG_1.xtc -n ../index.ndx -o ./rmsf_res_averaged.xvg -xvg none -res -b 5
        echo 18 | gmx_mpi rmsf -s ../5.equilibriation_second/npt.gro -f mdAlignCG_1.xtc -n ../index.ndx -o ./rmsf_res_whole_trj.xvg -xvg none -b 5
        
        echo finished job: $i
        
        cd ../..

    fi
done   




