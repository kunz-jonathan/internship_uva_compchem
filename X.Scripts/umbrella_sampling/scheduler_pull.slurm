#!/bin/bash
#SBATCH --job-name=pull
#SBATCH --output=smd_%j.log     
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --partition=gpuRTX-4th
#SBATCH --gpus-per-node=1
#SBATCH --array=1-4%4


#==========================#GROMACS,LOG, other configs etc.#==========================#


source /home/spack-user-2025/spack/share/spack/setup-env.sh
spack load gromacs@2024.3/qt


# no coredumps
ulimit -S -c 0
ulimit -s unlimited
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK


#==========================#SCHEDULER SCRIPT#==========================#


# Input folder containing files
INPUT_FOLDER="$SLURM_SUBMIT_DIR/0.Ligands"
OUTPUT_PARENT="$SLURM_SUBMIT_DIR"

#worker script
WORKER_SCRIPT="$SLURM_SUBMIT_DIR/../../../X.Scripts/binder_md_sim/worker_script.sh"

# Get all files in an array
FILES=($(ls "$INPUT_FOLDER"/* | sort -V))
NUM_FILES=${#FILES[@]}


# Safety check
if [ "$SLURM_ARRAY_TASK_ID" -ge "$NUM_FILES" ]; then
    echo "Task ID $SLURM_ARRAY_TASK_ID is out of range (0-$((NUM_FILES-1)))"
    exit 1
fi

echo "=== Starting Job $JOB_ID ==="
echo "Total files: $NUM_FILES"
echo "Processing files assigned to job $JOB_ID"


# Each job processes a subset of files in round-robin fashion
JOB_ID=$SLURM_ARRAY_TASK_ID


if [ "$SLURM_ARRAY_TASK_ID" -eq 1 ]; then

    mkdir -p "$OUTPUT_PARENT/jobs"
fi 

for ((i=JOB_ID; i<5; i+=4)); do 
    FILE="${FILES[$i]}"
    BASENAME=$(basename "$FILE")
    PREFIX="${BASENAME%%_*}"

    
    NEWFOLDER="$OUTPUT_PARENT/jobs/job_$PREFIX"
    mkdir -p "$NEWFOLDER"

    echo "[Job $JOB_ID] Copying $FILE → $NEWFOLDER/"
    cp "$FILE" "$NEWFOLDER/"

    echo "[Job $JOB_ID] Running worker script on $BASENAME"
    
    bash "$WORKER_SCRIPT" "$NEWFOLDER"
    cd ../..
done

echo "=== Finished Job $JOB_ID ==="




