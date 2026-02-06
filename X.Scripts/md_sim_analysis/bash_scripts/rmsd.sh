#!/bin/bash
#SBATCH --job-name=rmsd
#SBATCH --output=rmsd_%j.log     
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

        echo 17 18 | gmx_mpi rms -s ../5.equilibriation_second/npt.gro -f mdAlignCG_1.xtc -n ../index.ndx -o rmsd_whole_pro_to_whole_pep.xvg -xvg none -tu ns -b 5 -dt 1
        echo 19 18 | gmx_mpi rms -s ../5.equilibriation_second/npt.gro -f mdAlignCG_1.xtc -n ../index.ndx -o rmsd_CA_pro_to_whole_pep.xvg -xvg none -tu ns -b 5 -dt 1
           
        echo finished job: $i

        cd ../..

    fi
done   



