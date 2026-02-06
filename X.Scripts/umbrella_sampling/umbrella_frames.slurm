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


#####################################
# get_distances.sh
#
#   Script iteratively calls gmx distance and
#   assembles a series of COM distance values
#   indexed by frame number, for use in
#   preparing umbrella sampling windows.
#
# Written by: Justin A. Lemkul, Ph.D.
#    Contact: jalemkul@vt.edu


mkdir config_files

echo 0 | gmx_mpi trjconv -s ../6.pull/pull.tpr -f ../6.pull/pull.xtc -o config_files/conf.gro -sep

# make index file
gmx_mpi make_ndx -f ../6.pull/pull.tpr -o ./index.ndx <<EOF
ri 1-160 & a CA
ri 161-200 & ! r SOL & ! r NA & ! r CL
name 17 onlyProtein
name 18 onlyPeptide 
q
EOF

# compute distances for for the first 200 configurations which is usually enough to achieve a free energy landscape which reaches the unbound state of the binder

for (( i=0; i<200; i++ ))
do
    gmx_mpi distance -s ../6.pull/pull.tpr -f ./config_files/conf${i}.gro -n index.ndx -select 'com of group "onlyProtein" plus com of group "onlyPeptide"' -oall dist${i}.xvg 
done

# compile summary
touch summary_distances.dat
for (( i=0; i<200; i++ ))
do
    d=`tail -n 1 dist${i}.xvg | awk '{print $2}'`
    echo "${i} ${d}" >> summary_distances.dat
    rm dist${i}.xvg
done
#####################################

# file creation for frames sampled based on distance between the configurations
# values mainly based on experience and a more informed way of sampling could provide better performance in regards to computing time and landscape coverage
python3 /home/kunzj/internship_uva_compchem/X.Scripts/umbrella_sampling/helper_scripts/setupUmbrella.py summary_distances.dat 0.1 run_frame.sh &> caught-output.txt

# folder creation so that one frame has one folder
python3 /home/kunzj/internship_uva_compchem/X.Scripts/umbrella_sampling/helper_scripts/file_movement.py

# iterating now over every folder and performing the simulation
# identifier for folder


cd frames
dir_pattern="frame*"

# identifier for scripts
script_pattern="*_run_frame.sh"

# loop over matching directories

for dir in $dir_pattern; do
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